# src/causal_models.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import statsmodels.formula.api as smf
import statsmodels.api as sm
import logging

logger = logging.getLogger(__name__)


# ─── PROPENSITY SCORE MATCHING ────────────────────────────────────────────────

def compute_propensity_scores(
    user_df: pd.DataFrame,
    covariates: list[str],
) -> pd.DataFrame:
    """
    Fit a logistic regression to estimate P(treatment=1 | covariates).
    Returns user_df with a new 'propensity_score' column.
    """
    df = user_df.copy()
    X = df[covariates].fillna(df[covariates].median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, (df["variant"] == "treatment").astype(int))

    df["propensity_score"] = model.predict_proba(X_scaled)[:, 1]

    # Overlap / common support check
    ctrl_min = df.loc[df["variant"] == "control",    "propensity_score"].min()
    ctrl_max = df.loc[df["variant"] == "control",    "propensity_score"].max()
    trt_min  = df.loc[df["variant"] == "treatment",  "propensity_score"].min()
    trt_max  = df.loc[df["variant"] == "treatment",  "propensity_score"].max()
    overlap_min = max(ctrl_min, trt_min)
    overlap_max = min(ctrl_max, trt_max)

    outside = df[
        (df["propensity_score"] < overlap_min) |
        (df["propensity_score"] > overlap_max)
    ]
    if len(outside):
        logger.warning(
            f"Trimming {len(outside):,} units outside common support "
            f"[{overlap_min:.3f}, {overlap_max:.3f}]"
        )
        df = df[
            (df["propensity_score"] >= overlap_min) &
            (df["propensity_score"] <= overlap_max)
        ].copy()

    logger.info(
        f"Propensity scores computed — "
        f"mean control: {df.loc[df['variant']=='control','propensity_score'].mean():.3f}, "
        f"mean treatment: {df.loc[df['variant']=='treatment','propensity_score'].mean():.3f}"
    )
    return df


def nearest_neighbour_matching(
    df: pd.DataFrame,
    caliper: float = 0.05,
    ratio: int = 1,
    replacement: bool = False,
) -> pd.DataFrame:
    """
    1:ratio nearest-neighbour matching on propensity score within a caliper.

    caliper: max allowed propensity score distance (0.05 = standard rule-of-thumb)
    ratio:   how many controls to match per treated unit (1 = 1:1)
    replacement: whether controls can be reused across matches

    Returns a matched DataFrame with a 'match_id' column linking pairs.
    """
    treated  = df[df["variant"] == "treatment"].copy()
    controls = df[df["variant"] == "control"].copy()

    treated_ps  = treated[["propensity_score"]].values
    controls_ps = controls[["propensity_score"]].values

    nn = NearestNeighbors(n_neighbors=ratio, algorithm="ball_tree")
    nn.fit(controls_ps)
    distances, indices = nn.kneighbors(treated_ps)

    matched_rows = []
    used_control_indices = set()

    for i, (dists, idxs) in enumerate(zip(distances, indices)):
        treat_row = treated.iloc[i].copy()
        treat_row["match_id"] = i

        for dist, idx in zip(dists, idxs):
            if dist > caliper:
                continue
            if not replacement and idx in used_control_indices:
                continue
            ctrl_row = controls.iloc[idx].copy()
            ctrl_row["match_id"] = i
            matched_rows.append(treat_row)
            matched_rows.append(ctrl_row)
            if not replacement:
                used_control_indices.add(idx)
            break   # 1:1 — take first valid match only

    matched_df = pd.DataFrame(matched_rows).reset_index(drop=True)
    n_matched  = matched_df["match_id"].nunique()
    n_unmatched = len(treated) - n_matched

    logger.info(
        f"PSM complete — {n_matched:,} matched pairs "
        f"({n_unmatched:,} treated units unmatched, caliper={caliper})"
    )
    return matched_df


def psm_pipeline(
    user_df: pd.DataFrame,
    covariates: list[str],
    caliper: float = 0.05,
) -> pd.DataFrame:
    """Convenience wrapper: score → trim → match."""
    df_scored  = compute_propensity_scores(user_df, covariates)
    df_matched = nearest_neighbour_matching(df_scored, caliper=caliper)
    return df_matched


# ─── DIFFERENCE-IN-DIFFERENCES ────────────────────────────────────────────────

def difference_in_differences(
    df: pd.DataFrame,
    outcome: str = "avg_session_dur",
    pre_col: str = "pre_avg_session_dur",
) -> dict:
    """
    Classic 2x2 DiD estimator.

    Requires columns:
      - variant:  'control' | 'treatment'
      - {outcome}: post-period outcome
      - {pre_col}: pre-period outcome (same metric, before experiment)

    The DiD estimator is:
        ATT = (treat_post - treat_pre) - (ctrl_post - ctrl_pre)

    This differences out time-invariant unobserved confounders, provided the
    parallel trends assumption holds (tested via pre-period placebo below).
    """
    if pre_col not in df.columns:
        raise ValueError(
            f"DiD requires a pre-period column '{pre_col}'. "
            "Add pre-experiment session data or use PSM instead."
        )

    ctrl = df[df["variant"] == "control"]
    trt  = df[df["variant"] == "treatment"]

    delta_ctrl = ctrl[outcome].mean()  - ctrl[pre_col].mean()
    delta_trt  = trt[outcome].mean()   - trt[pre_col].mean()
    did_estimate = delta_trt - delta_ctrl
    pct_lift     = did_estimate / ctrl[pre_col].mean()

    logger.info(
        f"DiD — Δcontrol={delta_ctrl:.2f}s, Δtreatment={delta_trt:.2f}s, "
        f"ATT={did_estimate:.2f}s ({pct_lift:+.1%})"
    )
    return {
        "estimator":      "DiD",
        "delta_control":  delta_ctrl,
        "delta_treatment": delta_trt,
        "att":            did_estimate,
        "pct_lift":       pct_lift,
        "ctrl_pre_mean":  ctrl[pre_col].mean(),
        "ctrl_post_mean": ctrl[outcome].mean(),
        "trt_pre_mean":   trt[pre_col].mean(),
        "trt_post_mean":  trt[outcome].mean(),
    }


def did_regression(
    df: pd.DataFrame,
    outcome: str = "avg_session_dur",
    pre_col: str = "pre_avg_session_dur",
    covariates: list[str] | None = None,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Panel regression DiD with optional covariate adjustment.

    Stacks pre/post into long format and fits:
        Y = α + β1·post + β2·treated + β3·(post×treated) + γ·X + ε

    β3 is the DiD estimator (ATT). Standard errors are clustered by user_id.
    """
    if pre_col not in df.columns:
        raise ValueError(f"DiD regression requires pre-period column '{pre_col}'.")

    # Wide → long
    pre  = df[["user_id", "variant", pre_col]  + (covariates or [])].copy()
    post = df[["user_id", "variant", outcome]   + (covariates or [])].copy()
    pre["period"]  = 0
    post["period"] = 1
    pre  = pre.rename(columns={pre_col: "y"})
    post = post.rename(columns={outcome: "y"})
    panel = pd.concat([pre, post], ignore_index=True)

    panel["treated"]      = (panel["variant"] == "treatment").astype(int)
    panel["post"]         = panel["period"]
    panel["did_interact"] = panel["treated"] * panel["post"]

    covar_str = (" + " + " + ".join(covariates)) if covariates else ""
    formula   = f"y ~ post + treated + did_interact{covar_str}"

    model  = smf.ols(formula, data=panel).fit(
        cov_type="cluster",
        cov_kwds={"groups": panel["user_id"]}
    )
    logger.info(
        f"DiD regression — β(did_interact)={model.params['did_interact']:.3f}, "
        f"p={model.pvalues['did_interact']:.4e}, "
        f"95% CI=[{model.conf_int().loc['did_interact',0]:.3f}, "
        f"{model.conf_int().loc['did_interact',1]:.3f}]"
    )
    return model


# ─── MULTIVARIATE REGRESSION (OLS / ANCOVA) ───────────────────────────────────

def ancova(
    user_df: pd.DataFrame,
    outcome: str = "avg_session_dur",
    covariates: list[str] | None = None,
    log_outcome: bool = True,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    ANCOVA (OLS with covariate adjustment) on the post-period user frame.

    Using log(outcome) as the dependent variable is standard for session data:
      - normalises right-skewed residuals
      - makes the treatment coefficient directly interpretable as % lift

    Formula:  log(Y) = α + β·treated + γ1·X1 + γ2·X2 + ... + ε
    β is the estimated log-lift; exp(β) - 1 gives the % lift.
    """
    df = user_df.copy()

    if log_outcome:
        df["_outcome"] = np.log1p(df[outcome])
        y_col = "_outcome"
    else:
        y_col = outcome

    df["treated"] = (df["variant"] == "treatment").astype(int)

    covar_str = (" + " + " + ".join(covariates)) if covariates else ""
    formula   = f"{y_col} ~ treated{covar_str}"

    model = smf.ols(formula, data=df).fit(cov_type="HC3")  # heteroskedasticity-robust
    beta  = model.params["treated"]
    pct_lift = np.expm1(beta) if log_outcome else beta / df.loc[df["treated"]==0, y_col].mean()

    logger.info(
        f"ANCOVA — β(treated)={beta:.4f}, "
        f"pct_lift={pct_lift:+.1%}, "
        f"p={model.pvalues['treated']:.4e}, "
        f"R²={model.rsquared:.4f}"
    )
    return model


def summarise_model(
    model: sm.regression.linear_model.RegressionResultsWrapper,
    term: str = "treated",
    log_outcome: bool = True,
) -> dict:
    """Extract the key numbers for reporting."""
    beta = model.params[term]
    ci   = model.conf_int().loc[term]
    return {
        "estimator":  model.model.formula.split("~")[0].strip(),
        "term":       term,
        "coef":       beta,
        "pct_lift":   np.expm1(beta) if log_outcome else beta,
        "ci_low":     ci[0],
        "ci_high":    ci[1],
        "p_value":    model.pvalues[term],
        "n_obs":      int(model.nobs),
        "r_squared":  model.rsquared,
    }
