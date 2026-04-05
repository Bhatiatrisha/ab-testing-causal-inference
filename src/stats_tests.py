# src/stats_tests.py

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, bootstrap
import statsmodels.stats.proportion as smp
import logging

logger = logging.getLogger(__name__)


# ─── CORE SIGNIFICANCE TESTS ──────────────────────────────────────────────────

def welch_ttest(
    control: pd.Series,
    treatment: pd.Series,
    alpha: float = 0.01,
) -> dict:
    """
    Welch's t-test (unequal variances — always prefer over Student's).
    Returns effect size (Cohen's d), CI, and decision.
    """
    t, p = ttest_ind(control, treatment, equal_var=False, alternative="two-sided")

    # Cohen's d with pooled SD
    n_c, n_t   = len(control), len(treatment)
    mean_c, mean_t = control.mean(), treatment.mean()
    pooled_sd  = np.sqrt(
        ((n_c - 1) * control.std()**2 + (n_t - 1) * treatment.std()**2)
        / (n_c + n_t - 2)
    )
    cohens_d   = (mean_t - mean_c) / pooled_sd
    pct_lift   = (mean_t - mean_c) / mean_c

    # Confidence interval on the difference (Welch approximation)
    se = np.sqrt(control.var() / n_c + treatment.var() / n_t)
    df_welch = (
        (control.var()/n_c + treatment.var()/n_t)**2
        / ((control.var()/n_c)**2/(n_c-1) + (treatment.var()/n_t)**2/(n_t-1))
    )
    t_crit = stats.t.ppf(1 - alpha / 2, df=df_welch)
    diff   = mean_t - mean_c
    ci     = (diff - t_crit * se, diff + t_crit * se)

    result = {
        "test":         "Welch t-test",
        "mean_control": mean_c,
        "mean_treatment": mean_t,
        "pct_lift":     pct_lift,
        "cohens_d":     cohens_d,
        "t_stat":       t,
        "p_value":      p,
        "ci_low":       ci[0],
        "ci_high":      ci[1],
        "n_control":    n_c,
        "n_treatment":  n_t,
        "significant": bool(p < alpha),
        "alpha":        alpha,
    }
    _log_result(result)
    return result


def mann_whitney(
    control: pd.Series,
    treatment: pd.Series,
    alpha: float = 0.01,
) -> dict:
    """
    Mann-Whitney U test — non-parametric, robust to non-normality.
    Use as a complement to the t-test for highly skewed metrics.
    Reports rank-biserial correlation as effect size.
    """
    u, p = mannwhitneyu(control, treatment, alternative="two-sided")
    n_c, n_t = len(control), len(treatment)
    # Rank-biserial correlation (common language effect size)
    r = 1 - (2 * u) / (n_c * n_t)

    result = {
        "test":        "Mann-Whitney U",
        "u_stat":      u,
        "p_value":     p,
        "rank_biserial_r": r,
        "n_control":   n_c,
        "n_treatment": n_t,
        "significant": bool(p < alpha),
        "alpha":       alpha,
    }
    logger.info(
        f"Mann-Whitney U — U={u:.0f}, p={p:.4e}, r={r:.3f}, "
        f"{'SIGNIFICANT' if result['significant'] else 'not significant'} (α={alpha})"
    )
    return result


# ─── BOOTSTRAP CONFIDENCE INTERVAL ───────────────────────────────────────────

def bootstrap_mean_diff(
    control: pd.Series,
    treatment: pd.Series,
    n_resamples: int = 10_000,
    alpha: float = 0.01,
    random_state: int = 42,
) -> dict:
    """
    Percentile bootstrap CI on the mean difference (treatment - control).
    Model-free — makes no distributional assumptions.
    Complements the parametric Welch test.
    """
    rng = np.random.default_rng(random_state)
    ctrl_arr = control.to_numpy()
    trt_arr  = treatment.to_numpy()

    diffs = np.array([
        rng.choice(trt_arr,  len(trt_arr),  replace=True).mean() -
        rng.choice(ctrl_arr, len(ctrl_arr), replace=True).mean()
        for _ in range(n_resamples)
    ])

    ci_low, ci_high = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    point_est = treatment.mean() - control.mean()
    significant = bool(ci_low > 0 or ci_high < 0)  # CI excludes zero

    result = {
        "test":           "Bootstrap mean diff",
        "point_estimate": point_est,
        "pct_lift":       point_est / control.mean(),
        "ci_low":         ci_low,
        "ci_high":        ci_high,
        "n_resamples":    n_resamples,
        "significant":    significant,
        "alpha":          alpha,
    }
    logger.info(
        f"Bootstrap — diff={point_est:.3f} "
        f"[{ci_low:.3f}, {ci_high:.3f}], "
        f"{'SIGNIFICANT' if significant else 'not significant'} (α={alpha})"
    )
    return result


# ─── CUPED (VARIANCE REDUCTION) ──────────────────────────────────────────────

def cuped(
    user_df: pd.DataFrame,
    outcome_col: str    = "avg_session_dur",
    pre_period_col: str = "pre_avg_session_dur",
    alpha: float        = 0.01,
) -> dict:
    """
    CUPED — Controlled-experiment Using Pre-Experiment Data (Deng et al. 2013).

    Regresses the pre-experiment covariate out of the outcome to reduce variance,
    which increases statistical power without requiring more users.

        Y_cuped = Y - θ * (X - E[X])
        θ = Cov(Y, X) / Var(X)

    Runs a Welch t-test on the CUPED-adjusted outcome.
    Returns both raw and CUPED results for comparison.
    """
    if pre_period_col not in user_df.columns:
        raise ValueError(f"CUPED requires a pre-period column: '{pre_period_col}'")

    df = user_df.dropna(subset=[outcome_col, pre_period_col]).copy()

    # Estimate θ on the full dataset (pooled)
    cov_matrix = np.cov(df[outcome_col], df[pre_period_col])
    theta = cov_matrix[0, 1] / cov_matrix[1, 1]

    df["y_cuped"] = df[outcome_col] - theta * (df[pre_period_col] - df[pre_period_col].mean())

    ctrl_cuped = df.loc[df["variant"] == "control",    "y_cuped"]
    trt_cuped  = df.loc[df["variant"] == "treatment",  "y_cuped"]

    raw_result   = welch_ttest(
        df.loc[df["variant"]=="control",   outcome_col],
        df.loc[df["variant"]=="treatment", outcome_col],
        alpha=alpha
    )
    cuped_result = welch_ttest(ctrl_cuped, trt_cuped, alpha=alpha)

    var_reduction = 1 - ctrl_cuped.var() / df.loc[df["variant"]=="control", outcome_col].var()
    logger.info(f"CUPED variance reduction: {var_reduction:.1%}, θ={theta:.4f}")

    return {
        "theta":          theta,
        "var_reduction":  var_reduction,
        "raw":            raw_result,
        "cuped":          cuped_result,
    }


# ─── MULTIPLE TESTING CORRECTION ──────────────────────────────────────────────

def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """
    Benjamini-Hochberg FDR correction for multiple metrics/segments.
    Returns a boolean mask — True = reject H0 after correction.
    """
    m = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p   = np.array(p_values)[sorted_idx]
    thresholds = (np.arange(1, m + 1) / m) * alpha
    below      = sorted_p <= thresholds
    # Find the last True — all hypotheses up to that rank are rejected
    if not below.any():
        reject_sorted = np.zeros(m, dtype=bool)
    else:
        cutoff        = np.where(below)[0].max()
        reject_sorted = np.arange(m) <= cutoff
    # Restore original order
    reject = np.empty(m, dtype=bool)
    reject[sorted_idx] = reject_sorted
    logger.info(
        f"BH correction ({m} tests, α={alpha}) — "
        f"{reject.sum()} rejected, {m - reject.sum()} retained"
    )
    return reject.tolist()


# ─── FULL TEST BATTERY ────────────────────────────────────────────────────────

def run_significance_battery(
    user_df: pd.DataFrame,
    outcome_col: str    = "avg_session_dur",
    pre_period_col: str | None = None,
    alpha: float        = 0.01,
    n_bootstrap:  int   = 10_000,
) -> pd.DataFrame:
    """
    Run all applicable tests and return a tidy comparison DataFrame.
    Automatically adds CUPED if a pre-period column is available.
    """
    ctrl = user_df.loc[user_df["variant"] == "control",   outcome_col]
    trt  = user_df.loc[user_df["variant"] == "treatment",  outcome_col]

    rows = []

    # 1. Welch t-test
    r = welch_ttest(ctrl, trt, alpha=alpha)
    rows.append({
        "test":      r["test"],
        "pct_lift":  r["pct_lift"],
        "p_value":   r["p_value"],
        "ci_low":    r["ci_low"],
        "ci_high":   r["ci_high"],
        "significant": r["significant"],
        "effect_size": r["cohens_d"],
        "effect_metric": "Cohen's d",
    })

    # 2. Mann-Whitney U
    r = mann_whitney(ctrl, trt, alpha=alpha)
    rows.append({
        "test":      r["test"],
        "pct_lift":  None,
        "p_value":   r["p_value"],
        "ci_low":    None,
        "ci_high":   None,
        "significant": r["significant"],
        "effect_size": r["rank_biserial_r"],
        "effect_metric": "Rank-biserial r",
    })

    # 3. Bootstrap
    r = bootstrap_mean_diff(ctrl, trt, n_resamples=n_bootstrap, alpha=alpha)
    rows.append({
        "test":      r["test"],
        "pct_lift":  r["pct_lift"],
        "p_value":   None,
        "ci_low":    r["ci_low"],
        "ci_high":   r["ci_high"],
        "significant": r["significant"],
        "effect_size": r["point_estimate"],
        "effect_metric": "Mean diff (sec)",
    })

    # 4. CUPED (if pre-period available)
    if pre_period_col and pre_period_col in user_df.columns:
        r = cuped(user_df, outcome_col, pre_period_col, alpha=alpha)
        cr = r["cuped"]
        rows.append({
            "test":      f"CUPED t-test (var_red={r['var_reduction']:.1%})",
            "pct_lift":  cr["pct_lift"],
            "p_value":   cr["p_value"],
            "ci_low":    cr["ci_low"],
            "ci_high":   cr["ci_high"],
            "significant": cr["significant"],
            "effect_size": cr["cohens_d"],
            "effect_metric": "Cohen's d",
        })

    battery = pd.DataFrame(rows)
    logger.info(f"\n{battery.to_string(index=False)}")
    return battery


# ─── HELPER ───────────────────────────────────────────────────────────────────

def _log_result(r: dict) -> None:
    logger.info(
        f"{r['test']} — lift={r['pct_lift']:+.1%}, "
        f"d={r['cohens_d']:.3f}, p={r['p_value']:.4e}, "
        f"95% CI=[{r['ci_low']:.3f}, {r['ci_high']:.3f}], "
        f"{'SIGNIFICANT' if r['significant'] else 'not significant'} (α={r['alpha']})"
    )
