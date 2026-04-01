# src/power_analysis.py

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestIndPower, NormalIndPower
import statsmodels.stats.proportion as smp
import logging
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


# ─── DATA CLASSES ─────────────────────────────────────────────────────────────

@dataclass
class PowerResult:
    method:          str
    n_per_group:     int
    total_n:         int
    power:           float
    alpha:           float
    mde:             float          # minimum detectable effect (absolute)
    mde_pct:         float          # MDE as % of control mean
    effect_size:     float          # Cohen's d or equivalent
    baseline_mean:   float
    baseline_std:    float
    notes:           str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ─── CORE CALCULATORS ─────────────────────────────────────────────────────────

def required_sample_size(
    baseline_mean:    float,
    baseline_std:     float,
    mde_pct:          float  = 0.10,
    alpha:            float  = 0.05,
    power:            float  = 0.95,
    two_tailed:       bool   = True,
) -> PowerResult:
    """
    Compute required n-per-group for a two-sample t-test.

    baseline_mean: control group mean (e.g. avg session duration in seconds)
    baseline_std:  control group std  (from historical or pre-experiment data)
    mde_pct:       minimum detectable effect as fraction of baseline (e.g. 0.10 = 10%)
    alpha:         type I error rate
    power:         1 - type II error rate (0.95 = 95% power)
    """
    mde_abs    = baseline_mean * mde_pct
    cohens_d   = mde_abs / baseline_std
    alt        = "two-sided" if two_tailed else "larger"

    analysis   = TTestIndPower()
    n_per_group = analysis.solve_power(
        effect_size  = cohens_d,
        alpha        = alpha,
        power        = power,
        alternative  = alt,
    )
    n_per_group = int(np.ceil(n_per_group))

    result = PowerResult(
        method        = "two-sample t-test",
        n_per_group   = n_per_group,
        total_n       = n_per_group * 2,
        power         = power,
        alpha         = alpha,
        mde           = mde_abs,
        mde_pct       = mde_pct,
        effect_size   = cohens_d,
        baseline_mean = baseline_mean,
        baseline_std  = baseline_std,
    )
    logger.info(
        f"[t-test] n={n_per_group:,}/group (total={result.total_n:,}), "
        f"d={cohens_d:.4f}, MDE={mde_pct:.1%}, α={alpha}, power={power}"
    )
    return result


def required_sample_size_ancova(
    baseline_mean:  float,
    baseline_std:   float,
    r_squared:      float,
    mde_pct:        float = 0.10,
    alpha:          float = 0.05,
    power:          float = 0.95,
) -> PowerResult:
    """
    ANCOVA-adjusted sample size.

    ANCOVA with pre-experiment covariates reduces residual variance by the
    factor (1 - R²), where R² is the variance explained by the covariates.
    This directly shrinks the required n:

        n_ancova = n_ttest * (1 - R²)

    The R² here comes from regressing the outcome on your covariates
    (tenure_days, avg_pages_per_sess, etc.) — use model.rsquared from
    the ANCOVA fit in causal_models.py.
    """
    base_result = required_sample_size(
        baseline_mean, baseline_std, mde_pct, alpha, power
    )

    # Variance reduction factor
    variance_reduction = 1 - r_squared
    n_per_group = int(np.ceil(base_result.n_per_group * variance_reduction))
    pct_reduction = 1 - (n_per_group / base_result.n_per_group)

    result = PowerResult(
        method        = "ANCOVA (covariate-adjusted)",
        n_per_group   = n_per_group,
        total_n       = n_per_group * 2,
        power         = power,
        alpha         = alpha,
        mde           = base_result.mde,
        mde_pct       = mde_pct,
        effect_size   = base_result.effect_size,
        baseline_mean = baseline_mean,
        baseline_std  = baseline_std,
        notes         = (
            f"R²={r_squared:.3f} → variance reduction={1-variance_reduction:.1%}, "
            f"sample reduction vs t-test={pct_reduction:.1%}"
        )
    )
    logger.info(
        f"[ANCOVA] n={n_per_group:,}/group (total={result.total_n:,}), "
        f"R²={r_squared:.3f}, sample reduction={pct_reduction:.1%}"
    )
    return result


def achieved_power(
    n_per_group:   int,
    baseline_mean: float,
    baseline_std:  float,
    mde_pct:       float = 0.10,
    alpha:         float = 0.05,
) -> float:
    """
    Given a fixed n already collected, compute the achieved power.
    Useful for post-hoc reporting — 'we had X% power to detect Y% lift'.
    """
    mde_abs  = baseline_mean * mde_pct
    cohens_d = mde_abs / baseline_std
    analysis = TTestIndPower()
    pwr = analysis.solve_power(
        effect_size = cohens_d,
        nobs1       = n_per_group,
        alpha       = alpha,
    )
    logger.info(
        f"Achieved power={pwr:.3f} at n={n_per_group:,}/group, "
        f"d={cohens_d:.4f}, α={alpha}"
    )
    return float(pwr)


def minimum_detectable_effect(
    n_per_group:   int,
    baseline_mean: float,
    baseline_std:  float,
    alpha:         float = 0.05,
    power:         float = 0.95,
) -> dict:
    """
    Given a fixed n, compute the smallest effect that is detectable
    at the given alpha and power. Returns both absolute and % MDE.
    """
    analysis = TTestIndPower()
    cohens_d = analysis.solve_power(
        nobs1 = n_per_group,
        alpha = alpha,
        power = power,
    )
    mde_abs = cohens_d * baseline_std
    mde_pct = mde_abs / baseline_mean

    logger.info(
        f"MDE at n={n_per_group:,}/group: {mde_pct:.1%} "
        f"({mde_abs:.2f}s abs), d={cohens_d:.4f}"
    )
    return {
        "n_per_group":   n_per_group,
        "cohens_d":      cohens_d,
        "mde_abs":       mde_abs,
        "mde_pct":       mde_pct,
        "baseline_mean": baseline_mean,
        "baseline_std":  baseline_std,
        "alpha":         alpha,
        "power":         power,
    }


# ─── SENSITIVITY ANALYSIS ─────────────────────────────────────────────────────

def power_curve(
    baseline_mean:  float,
    baseline_std:   float,
    mde_pcts:       list[float] | None = None,
    alpha_levels:   list[float] | None = None,
    power:          float = 0.95,
    r_squared:      float | None = None,
) -> pd.DataFrame:
    """
    Sweep over MDE values and optionally alpha levels to build a sensitivity
    table. If r_squared is provided, adds ANCOVA-adjusted columns.

    Returns a DataFrame suitable for plotting or inclusion in the report.
    """
    if mde_pcts is None:
        mde_pcts = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]
    if alpha_levels is None:
        alpha_levels = [0.05, 0.01]

    rows = []
    for mde in mde_pcts:
        for alpha in alpha_levels:
            base = required_sample_size(
                baseline_mean, baseline_std, mde, alpha, power
            )
            row = {
                "mde_pct":         mde,
                "alpha":           alpha,
                "n_ttest":         base.n_per_group,
                "total_ttest":     base.total_n,
            }
            if r_squared is not None:
                anc = required_sample_size_ancova(
                    baseline_mean, baseline_std, r_squared, mde, alpha, power
                )
                row["n_ancova"]       = anc.n_per_group
                row["total_ancova"]   = anc.total_n
                row["pct_reduction"]  = 1 - anc.n_per_group / base.n_per_group
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def runtime_estimate(
    required_n:      int,
    daily_traffic:   int,
    treatment_split: float = 0.5,
) -> dict:
    """
    Estimate how many days the experiment needs to run.
    Accounts for the fraction of traffic allocated to the experiment.
    """
    daily_per_group  = daily_traffic * treatment_split
    days_required    = required_n / daily_per_group
    weeks            = days_required / 7

    logger.info(
        f"Runtime: {days_required:.1f} days ({weeks:.1f} weeks) "
        f"at {daily_traffic:,} users/day, {treatment_split:.0%} split"
    )
    return {
        "required_n_per_group": required_n,
        "daily_traffic":        daily_traffic,
        "treatment_split":      treatment_split,
        "days_required":        round(days_required, 1),
        "weeks_required":       round(weeks, 1),
    }


# ─── FULL ANALYSIS FROM OBSERVED DATA ─────────────────────────────────────────

def auto_power_analysis(
    user_df:    pd.DataFrame,
    outcome:    str   = "avg_session_dur",
    mde_pct:    float = 0.10,
    alpha:      float = 0.01,
    power:      float = 0.95,
    r_squared:  float | None = None,
) -> dict:
    """
    Derive all inputs directly from observed data and run the full battery.
    Pass r_squared from your ANCOVA model (model.rsquared) to get the
    ANCOVA-adjusted comparison and sample size reduction.
    """
    ctrl = user_df.loc[user_df["variant"] == "control", outcome].dropna()
    baseline_mean = ctrl.mean()
    baseline_std  = ctrl.std()
    n_per_group   = len(ctrl)

    logger.info(
        f"Auto power analysis — baseline: mean={baseline_mean:.2f}, "
        f"std={baseline_std:.2f}, n_control={n_per_group:,}"
    )

    ttest_req  = required_sample_size(baseline_mean, baseline_std, mde_pct, alpha, power)
    ach_power  = achieved_power(n_per_group, baseline_mean, baseline_std, mde_pct, alpha)
    mde_info   = minimum_detectable_effect(n_per_group, baseline_mean, baseline_std, alpha, power)
    curve      = power_curve(baseline_mean, baseline_std, r_squared=r_squared)

    results = {
        "baseline_mean":     baseline_mean,
        "baseline_std":      baseline_std,
        "n_observed":        n_per_group,
        "ttest_requirement": ttest_req,
        "achieved_power":    ach_power,
        "mde_at_n":          mde_info,
        "sensitivity_table": curve,
    }

    if r_squared is not None:
        ancova_req = required_sample_size_ancova(
            baseline_mean, baseline_std, r_squared, mde_pct, alpha, power
        )
        pct_reduction = 1 - ancova_req.n_per_group / ttest_req.n_per_group
        results["ancova_requirement"]    = ancova_req
        results["sample_size_reduction"] = pct_reduction
        logger.info(f"Sample size reduction (ANCOVA vs t-test): {pct_reduction:.1%}")

    return results
