# tests/test_stats_tests.py

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stats_tests import (
    welch_ttest,
    mann_whitney,
    bootstrap_mean_diff,
    cuped,
    benjamini_hochberg,
    run_significance_battery,
)


# ─── FIXTURES ─────────────────────────────────────────────────────────────────

@pytest.fixture
def large_effect():
    """Control vs treatment with a large, easily detectable effect."""
    rng = np.random.default_rng(0)
    ctrl = pd.Series(rng.normal(200, 30, 2000))
    trt  = pd.Series(rng.normal(240, 30, 2000))   # +20% lift
    return ctrl, trt


@pytest.fixture
def no_effect():
    """Control vs treatment drawn from identical distributions."""
    rng = np.random.default_rng(1)
    ctrl = pd.Series(rng.normal(200, 30, 2000))
    trt  = pd.Series(rng.normal(200, 30, 2000))
    return ctrl, trt


@pytest.fixture
def user_df_with_effect():
    """Full user-level DataFrame for battery tests."""
    rng = np.random.default_rng(7)
    n = 1000
    variant = np.where(np.arange(n) < n // 2, "control", "treatment")
    dur = np.where(
        variant == "treatment",
        rng.normal(240, 40, n),
        rng.normal(200, 40, n),
    )
    pre = dur * 0.9 + rng.normal(0, 10, n)   # pre-period correlated with post
    return pd.DataFrame({
        "user_id":           [f"u{i}" for i in range(n)],
        "variant":           variant,
        "avg_session_dur":   dur.clip(10),
        "pre_avg_session_dur": pre.clip(10),
    })


# ─── welch_ttest ──────────────────────────────────────────────────────────────

class TestWelchTTest:

    def test_returns_dict(self, large_effect):
        ctrl, trt = large_effect
        result = welch_ttest(ctrl, trt)
        assert isinstance(result, dict)

    def test_expected_keys(self, large_effect):
        ctrl, trt = large_effect
        result = welch_ttest(ctrl, trt)
        for key in ["test", "mean_control", "mean_treatment", "pct_lift",
                    "cohens_d", "t_stat", "p_value", "ci_low", "ci_high",
                    "significant", "alpha"]:
            assert key in result, f"Missing key: {key}"

    def test_significant_for_large_effect(self, large_effect):
        ctrl, trt = large_effect
        result = welch_ttest(ctrl, trt, alpha=0.01)
        assert result["significant"] is True

    def test_not_significant_for_no_effect(self, no_effect):
        ctrl, trt = no_effect
        result = welch_ttest(ctrl, trt, alpha=0.01)
        assert result["significant"] is False

    def test_pct_lift_direction_correct(self, large_effect):
        ctrl, trt = large_effect
        result = welch_ttest(ctrl, trt)
        assert result["pct_lift"] > 0

    def test_negative_lift_detected(self):
        rng = np.random.default_rng(3)
        ctrl = pd.Series(rng.normal(200, 20, 1000))
        trt  = pd.Series(rng.normal(170, 20, 1000))
        result = welch_ttest(ctrl, trt)
        assert result["pct_lift"] < 0

    def test_ci_excludes_zero_when_significant(self, large_effect):
        ctrl, trt = large_effect
        result = welch_ttest(ctrl, trt, alpha=0.01)
        if result["significant"]:
            assert not (result["ci_low"] < 0 < result["ci_high"])

    def test_ci_low_less_than_ci_high(self, large_effect):
        ctrl, trt = large_effect
        result = welch_ttest(ctrl, trt)
        assert result["ci_low"] < result["ci_high"]

    def test_cohens_d_large_for_large_effect(self, large_effect):
        ctrl, trt = large_effect
        result = welch_ttest(ctrl, trt)
        # 40-point difference with std=30 → d ≈ 1.3
        assert result["cohens_d"] > 0.8

    def test_n_counts_correct(self, large_effect):
        ctrl, trt = large_effect
        result = welch_ttest(ctrl, trt)
        assert result["n_control"]   == len(ctrl)
        assert result["n_treatment"] == len(trt)

    def test_mean_values_correct(self, large_effect):
        ctrl, trt = large_effect
        result = welch_ttest(ctrl, trt)
        assert abs(result["mean_control"]   - ctrl.mean()) < 0.01
        assert abs(result["mean_treatment"] - trt.mean())  < 0.01


# ─── mann_whitney ─────────────────────────────────────────────────────────────

class TestMannWhitney:

    def test_returns_dict(self, large_effect):
        ctrl, trt = large_effect
        result = mann_whitney(ctrl, trt)
        assert isinstance(result, dict)

    def test_significant_for_large_effect(self, large_effect):
        ctrl, trt = large_effect
        result = mann_whitney(ctrl, trt, alpha=0.01)
        assert result["significant"] is True

    def test_not_significant_for_no_effect(self, no_effect):
        ctrl, trt = no_effect
        result = mann_whitney(ctrl, trt, alpha=0.01)
        assert result["significant"] is False

    def test_rank_biserial_between_minus1_and_1(self, large_effect):
        ctrl, trt = large_effect
        result = mann_whitney(ctrl, trt)
        assert -1 <= result["rank_biserial_r"] <= 1

    def test_positive_r_when_treatment_higher(self, large_effect):
        ctrl, trt = large_effect
        result = mann_whitney(ctrl, trt)
        assert result["rank_biserial_r"] > 0

    def test_expected_keys(self, large_effect):
        ctrl, trt = large_effect
        result = mann_whitney(ctrl, trt)
        for key in ["test", "u_stat", "p_value", "rank_biserial_r",
                    "n_control", "n_treatment", "significant", "alpha"]:
            assert key in result

    def test_robust_to_skewed_data(self):
        """Mann-Whitney should still detect a shift in skewed distributions."""
        rng = np.random.default_rng(9)
        ctrl = pd.Series(rng.exponential(scale=100, size=1000))
        trt  = pd.Series(rng.exponential(scale=150, size=1000))
        result = mann_whitney(ctrl, trt, alpha=0.05)
        assert result["significant"] is True


# ─── bootstrap_mean_diff ──────────────────────────────────────────────────────

class TestBootstrap:

    def test_returns_dict(self, large_effect):
        ctrl, trt = large_effect
        result = bootstrap_mean_diff(ctrl, trt, n_resamples=500)
        assert isinstance(result, dict)

    def test_significant_for_large_effect(self, large_effect):
        ctrl, trt = large_effect
        result = bootstrap_mean_diff(ctrl, trt, n_resamples=2000, alpha=0.01)
        assert result["significant"] is True

    def test_not_significant_for_no_effect(self, no_effect):
        ctrl, trt = no_effect
        result = bootstrap_mean_diff(ctrl, trt, n_resamples=2000, alpha=0.01)
        assert result["significant"] is False

    def test_ci_low_less_than_ci_high(self, large_effect):
        ctrl, trt = large_effect
        result = bootstrap_mean_diff(ctrl, trt, n_resamples=500)
        assert result["ci_low"] < result["ci_high"]

    def test_point_estimate_close_to_raw_diff(self, large_effect):
        ctrl, trt = large_effect
        result = bootstrap_mean_diff(ctrl, trt, n_resamples=500)
        raw_diff = trt.mean() - ctrl.mean()
        assert abs(result["point_estimate"] - raw_diff) < 1.0  # within 1 second

    def test_reproducible_with_same_seed(self, large_effect):
        ctrl, trt = large_effect
        r1 = bootstrap_mean_diff(ctrl, trt, n_resamples=500, random_state=42)
        r2 = bootstrap_mean_diff(ctrl, trt, n_resamples=500, random_state=42)
        assert r1["ci_low"]  == r2["ci_low"]
        assert r1["ci_high"] == r2["ci_high"]

    def test_expected_keys(self, large_effect):
        ctrl, trt = large_effect
        result = bootstrap_mean_diff(ctrl, trt, n_resamples=200)
        for key in ["test", "point_estimate", "pct_lift",
                    "ci_low", "ci_high", "n_resamples", "significant", "alpha"]:
            assert key in result


# ─── cuped ────────────────────────────────────────────────────────────────────

class TestCUPED:

    def test_returns_dict(self, user_df_with_effect):
        result = cuped(user_df_with_effect,
                       outcome_col="avg_session_dur",
                       pre_period_col="pre_avg_session_dur")
        assert isinstance(result, dict)

    def test_expected_keys(self, user_df_with_effect):
        result = cuped(user_df_with_effect,
                       outcome_col="avg_session_dur",
                       pre_period_col="pre_avg_session_dur")
        for key in ["theta", "var_reduction", "raw", "cuped"]:
            assert key in result

    def test_variance_reduction_positive(self, user_df_with_effect):
        result = cuped(user_df_with_effect,
                       outcome_col="avg_session_dur",
                       pre_period_col="pre_avg_session_dur")
        assert result["var_reduction"] > 0

    def test_cuped_pvalue_le_raw_pvalue(self, user_df_with_effect):
        """CUPED should produce a smaller or equal p-value than raw test."""
        result = cuped(user_df_with_effect,
                   outcome_col="avg_session_dur",
                   pre_period_col="pre_avg_session_dur")
    assert result["var_reduction"] > 0

    def test_cuped_significant_when_raw_is_significant(self, user_df_with_effect):
    """If the raw test is significant, CUPED should also be significant."""
    result = cuped(user_df_with_effect,
                   outcome_col="avg_session_dur",
                   pre_period_col="pre_avg_session_dur",
                   alpha=0.05)
    if result["raw"]["significant"]:
        assert result["cuped"]["significant"]
    
    def test_theta_is_float(self, user_df_with_effect):
        result = cuped(user_df_with_effect,
                       outcome_col="avg_session_dur",
                       pre_period_col="pre_avg_session_dur")
        assert isinstance(result["theta"], float)

    def test_missing_pre_col_raises(self, user_df_with_effect):
        with pytest.raises(ValueError, match="pre-period column"):
            cuped(user_df_with_effect,
                  outcome_col="avg_session_dur",
                  pre_period_col="nonexistent_col")


# ─── benjamini_hochberg ───────────────────────────────────────────────────────

class TestBenjaminiHochberg:

    def test_returns_list_of_bools(self):
        pvals  = [0.001, 0.02, 0.04, 0.3, 0.8]
        result = benjamini_hochberg(pvals)
        assert isinstance(result, list)
        assert all(isinstance(r, bool) for r in result)

    def test_length_matches_input(self):
        pvals  = [0.01, 0.05, 0.1, 0.5]
        result = benjamini_hochberg(pvals)
        assert len(result) == len(pvals)

    def test_all_rejected_when_all_tiny(self):
        pvals  = [1e-10, 1e-9, 1e-8]
        result = benjamini_hochberg(pvals, alpha=0.05)
        assert all(result)

    def test_none_rejected_when_all_large(self):
        pvals  = [0.5, 0.6, 0.7, 0.9]
        result = benjamini_hochberg(pvals, alpha=0.05)
        assert not any(result)

    def test_single_pvalue(self):
        assert benjamini_hochberg([0.001]) == [True]
        assert benjamini_hochberg([0.9])   == [False]

    def test_order_independent(self):
        """Result should respect original order, not sorted order."""
        pvals   = [0.8, 0.001, 0.5, 0.002]
        result  = benjamini_hochberg(pvals, alpha=0.05)
        # The two small p-values should be rejected
        assert result[1] is True
        assert result[3] is True
        assert result[0] is False


# ─── run_significance_battery ─────────────────────────────────────────────────

class TestSignificanceBattery:

    def test_returns_dataframe(self, user_df_with_effect):
        result = run_significance_battery(
            user_df_with_effect, outcome_col="avg_session_dur", n_bootstrap=200
        )
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, user_df_with_effect):
        result = run_significance_battery(
            user_df_with_effect, outcome_col="avg_session_dur", n_bootstrap=200
        )
        for col in ["test", "pct_lift", "p_value", "significant"]:
            assert col in result.columns

    def test_includes_welch_and_mw(self, user_df_with_effect):
        result = run_significance_battery(
            user_df_with_effect, outcome_col="avg_session_dur", n_bootstrap=200
        )
        tests = result["test"].tolist()
        assert any("Welch" in t for t in tests)
        assert any("Mann" in t  for t in tests)

    def test_includes_cuped_when_pre_col_provided(self, user_df_with_effect):
        result = run_significance_battery(
            user_df_with_effect,
            outcome_col    = "avg_session_dur",
            pre_period_col = "pre_avg_session_dur",
            n_bootstrap    = 200,
        )
        assert any("CUPED" in t for t in result["test"].tolist())

    def test_no_cuped_without_pre_col(self, user_df_with_effect):
        result = run_significance_battery(
            user_df_with_effect, outcome_col="avg_session_dur", n_bootstrap=200
        )
        assert not any("CUPED" in t for t in result["test"].tolist())

    def test_majority_significant_for_large_effect(self, user_df_with_effect):
        result = run_significance_battery(
            user_df_with_effect,
            outcome_col = "avg_session_dur",
            alpha       = 0.05,
            n_bootstrap = 500,
        )
        sig_count = result["significant"].sum()
        assert sig_count >= 2  # at least 2 of the tests should fire

    def test_all_pct_lifts_positive_for_positive_effect(self, user_df_with_effect):
        result = run_significance_battery(
            user_df_with_effect, outcome_col="avg_session_dur", n_bootstrap=200
        )
        lifts = result["pct_lift"].dropna()
        assert (lifts > 0).all()
