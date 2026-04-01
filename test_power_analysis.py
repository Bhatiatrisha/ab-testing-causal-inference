# tests/test_power_analysis.py

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.power_analysis import (
    required_sample_size,
    required_sample_size_ancova,
    achieved_power,
    minimum_detectable_effect,
    power_curve,
    runtime_estimate,
    auto_power_analysis,
    PowerResult,
)


# ─── FIXTURES ─────────────────────────────────────────────────────────────────

BASELINE_MEAN = 200.0   # seconds
BASELINE_STD  = 80.0    # seconds  (CV ≈ 0.4, realistic for session data)

@pytest.fixture
def user_df():
    """Simple user-level DataFrame for auto_power_analysis tests."""
    rng = np.random.default_rng(42)
    n   = 2000
    variant = np.where(np.arange(n) < n // 2, "control", "treatment")
    dur = np.where(
        variant == "treatment",
        rng.normal(BASELINE_MEAN * 1.12, BASELINE_STD, n),
        rng.normal(BASELINE_MEAN,        BASELINE_STD, n),
    )
    return pd.DataFrame({
        "user_id":         [f"u{i}" for i in range(n)],
        "variant":         variant,
        "avg_session_dur": dur.clip(10),
    })


# ─── PowerResult dataclass ────────────────────────────────────────────────────

class TestPowerResult:

    def test_to_dict_returns_dict(self):
        r = PowerResult(
            method="test", n_per_group=100, total_n=200,
            power=0.95, alpha=0.05, mde=20.0, mde_pct=0.10,
            effect_size=0.25, baseline_mean=200.0, baseline_std=80.0,
        )
        d = r.to_dict()
        assert isinstance(d, dict)
        assert d["n_per_group"] == 100

    def test_total_n_is_double_n_per_group(self):
        r = PowerResult(
            method="t-test", n_per_group=500, total_n=1000,
            power=0.95, alpha=0.01, mde=20.0, mde_pct=0.10,
            effect_size=0.25, baseline_mean=200.0, baseline_std=80.0,
        )
        assert r.total_n == r.n_per_group * 2


# ─── required_sample_size ────────────────────────────────────────────────────

class TestRequiredSampleSize:

    def test_returns_power_result(self):
        result = required_sample_size(BASELINE_MEAN, BASELINE_STD)
        assert isinstance(result, PowerResult)

    def test_n_per_group_positive_integer(self):
        result = required_sample_size(BASELINE_MEAN, BASELINE_STD)
        assert result.n_per_group > 0
        assert isinstance(result.n_per_group, int)

    def test_total_n_double_n_per_group(self):
        result = required_sample_size(BASELINE_MEAN, BASELINE_STD)
        assert result.total_n == result.n_per_group * 2

    def test_larger_mde_needs_fewer_users(self):
        r_small = required_sample_size(BASELINE_MEAN, BASELINE_STD, mde_pct=0.05)
        r_large = required_sample_size(BASELINE_MEAN, BASELINE_STD, mde_pct=0.20)
        assert r_large.n_per_group < r_small.n_per_group

    def test_stricter_alpha_needs_more_users(self):
        r_loose  = required_sample_size(BASELINE_MEAN, BASELINE_STD, alpha=0.10)
        r_strict = required_sample_size(BASELINE_MEAN, BASELINE_STD, alpha=0.01)
        assert r_strict.n_per_group > r_loose.n_per_group

    def test_higher_power_needs_more_users(self):
        r_low  = required_sample_size(BASELINE_MEAN, BASELINE_STD, power=0.80)
        r_high = required_sample_size(BASELINE_MEAN, BASELINE_STD, power=0.95)
        assert r_high.n_per_group > r_low.n_per_group

    def test_higher_std_needs_more_users(self):
        r_tight = required_sample_size(BASELINE_MEAN, 40.0)
        r_wide  = required_sample_size(BASELINE_MEAN, 120.0)
        assert r_wide.n_per_group > r_tight.n_per_group

    def test_mde_abs_correct(self):
        result = required_sample_size(BASELINE_MEAN, BASELINE_STD, mde_pct=0.10)
        assert abs(result.mde - BASELINE_MEAN * 0.10) < 0.01

    def test_effect_size_positive(self):
        result = required_sample_size(BASELINE_MEAN, BASELINE_STD)
        assert result.effect_size > 0

    def test_metadata_stored(self):
        result = required_sample_size(
            BASELINE_MEAN, BASELINE_STD, mde_pct=0.12, alpha=0.01, power=0.95
        )
        assert result.alpha        == 0.01
        assert result.power        == 0.95
        assert result.mde_pct      == 0.12
        assert result.baseline_mean == BASELINE_MEAN


# ─── required_sample_size_ancova ─────────────────────────────────────────────

class TestANCOVASampleSize:

    def test_returns_power_result(self):
        result = required_sample_size_ancova(BASELINE_MEAN, BASELINE_STD, r_squared=0.3)
        assert isinstance(result, PowerResult)

    def test_ancova_needs_fewer_users_than_ttest(self):
        base   = required_sample_size(BASELINE_MEAN, BASELINE_STD, mde_pct=0.10)
        anc    = required_sample_size_ancova(BASELINE_MEAN, BASELINE_STD,
                                             r_squared=0.3, mde_pct=0.10)
        assert anc.n_per_group < base.n_per_group

    def test_reduction_proportional_to_r_squared(self):
        base  = required_sample_size(BASELINE_MEAN, BASELINE_STD)
        r2    = 0.25
        anc   = required_sample_size_ancova(BASELINE_MEAN, BASELINE_STD, r_squared=r2)
        expected_n = int(np.ceil(base.n_per_group * (1 - r2)))
        assert anc.n_per_group == expected_n

    def test_higher_r_squared_gives_more_reduction(self):
        anc_low  = required_sample_size_ancova(BASELINE_MEAN, BASELINE_STD, r_squared=0.10)
        anc_high = required_sample_size_ancova(BASELINE_MEAN, BASELINE_STD, r_squared=0.50)
        assert anc_high.n_per_group < anc_low.n_per_group

    def test_r_squared_zero_same_as_ttest(self):
        base = required_sample_size(BASELINE_MEAN, BASELINE_STD)
        anc  = required_sample_size_ancova(BASELINE_MEAN, BASELINE_STD, r_squared=0.0)
        assert anc.n_per_group == base.n_per_group

    def test_notes_field_contains_reduction_info(self):
        anc = required_sample_size_ancova(BASELINE_MEAN, BASELINE_STD, r_squared=0.25)
        assert "reduction" in anc.notes.lower() or "R²" in anc.notes

    def test_22pct_reduction_at_r2_022(self):
        """Reproduce the headline project result."""
        base = required_sample_size(BASELINE_MEAN, BASELINE_STD,
                                    mde_pct=0.10, alpha=0.01, power=0.95)
        anc  = required_sample_size_ancova(BASELINE_MEAN, BASELINE_STD,
                                           r_squared=0.22,
                                           mde_pct=0.10, alpha=0.01, power=0.95)
        actual_reduction = 1 - anc.n_per_group / base.n_per_group
        assert abs(actual_reduction - 0.22) < 0.02   # within 2pp of 22%


# ─── achieved_power ───────────────────────────────────────────────────────────

class TestAchievedPower:

    def test_returns_float(self):
        pwr = achieved_power(500, BASELINE_MEAN, BASELINE_STD)
        assert isinstance(pwr, float)

    def test_between_0_and_1(self):
        pwr = achieved_power(500, BASELINE_MEAN, BASELINE_STD)
        assert 0 <= pwr <= 1

    def test_larger_n_gives_higher_power(self):
        p_small = achieved_power(200,  BASELINE_MEAN, BASELINE_STD)
        p_large = achieved_power(5000, BASELINE_MEAN, BASELINE_STD)
        assert p_large > p_small

    def test_approaches_1_for_very_large_n(self):
        pwr = achieved_power(100_000, BASELINE_MEAN, BASELINE_STD, mde_pct=0.05)
        assert pwr > 0.999

    def test_near_zero_for_tiny_n_small_effect(self):
        pwr = achieved_power(10, BASELINE_MEAN, BASELINE_STD, mde_pct=0.01)
        assert pwr < 0.20

    def test_stricter_alpha_lowers_power(self):
        p_loose  = achieved_power(500, BASELINE_MEAN, BASELINE_STD, alpha=0.10)
        p_strict = achieved_power(500, BASELINE_MEAN, BASELINE_STD, alpha=0.001)
        assert p_strict < p_loose


# ─── minimum_detectable_effect ───────────────────────────────────────────────

class TestMDE:

    def test_returns_dict(self):
        result = minimum_detectable_effect(500, BASELINE_MEAN, BASELINE_STD)
        assert isinstance(result, dict)

    def test_expected_keys(self):
        result = minimum_detectable_effect(500, BASELINE_MEAN, BASELINE_STD)
        for key in ["n_per_group", "cohens_d", "mde_abs", "mde_pct",
                    "baseline_mean", "baseline_std", "alpha", "power"]:
            assert key in result

    def test_mde_pct_positive(self):
        result = minimum_detectable_effect(500, BASELINE_MEAN, BASELINE_STD)
        assert result["mde_pct"] > 0

    def test_larger_n_gives_smaller_mde(self):
        r_small = minimum_detectable_effect(200,   BASELINE_MEAN, BASELINE_STD)
        r_large = minimum_detectable_effect(10000, BASELINE_MEAN, BASELINE_STD)
        assert r_large["mde_pct"] < r_small["mde_pct"]

    def test_mde_abs_consistent_with_mde_pct(self):
        result = minimum_detectable_effect(500, BASELINE_MEAN, BASELINE_STD)
        expected_abs = result["mde_pct"] * BASELINE_MEAN
        assert abs(result["mde_abs"] - expected_abs) < 0.01

    def test_metadata_stored_correctly(self):
        result = minimum_detectable_effect(500, BASELINE_MEAN, BASELINE_STD,
                                           alpha=0.01, power=0.95)
        assert result["n_per_group"]   == 500
        assert result["baseline_mean"] == BASELINE_MEAN
        assert result["alpha"]         == 0.01
        assert result["power"]         == 0.95


# ─── power_curve ─────────────────────────────────────────────────────────────

class TestPowerCurve:

    def test_returns_dataframe(self):
        df = power_curve(BASELINE_MEAN, BASELINE_STD)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns_without_r_squared(self):
        df = power_curve(BASELINE_MEAN, BASELINE_STD)
        for col in ["mde_pct", "alpha", "n_ttest", "total_ttest"]:
            assert col in df.columns

    def test_ancova_columns_added_with_r_squared(self):
        df = power_curve(BASELINE_MEAN, BASELINE_STD, r_squared=0.25)
        for col in ["n_ancova", "total_ancova", "pct_reduction"]:
            assert col in df.columns

    def test_row_count_matches_mde_alpha_product(self):
        mdes   = [0.05, 0.10, 0.15]
        alphas = [0.05, 0.01]
        df = power_curve(BASELINE_MEAN, BASELINE_STD,
                         mde_pcts=mdes, alpha_levels=alphas)
        assert len(df) == len(mdes) * len(alphas)

    def test_n_ttest_decreases_as_mde_increases(self):
        df = power_curve(BASELINE_MEAN, BASELINE_STD,
                         mde_pcts=[0.05, 0.10, 0.20], alpha_levels=[0.05])
        ns = df["n_ttest"].tolist()
        assert ns[0] > ns[1] > ns[2]

    def test_pct_reduction_between_0_and_1(self):
        df = power_curve(BASELINE_MEAN, BASELINE_STD, r_squared=0.30)
        assert (df["pct_reduction"].between(0, 1)).all()

    def test_all_ns_positive(self):
        df = power_curve(BASELINE_MEAN, BASELINE_STD)
        assert (df["n_ttest"] > 0).all()


# ─── runtime_estimate ─────────────────────────────────────────────────────────

class TestRuntimeEstimate:

    def test_returns_dict(self):
        result = runtime_estimate(1000, 500)
        assert isinstance(result, dict)

    def test_expected_keys(self):
        result = runtime_estimate(1000, 500)
        for key in ["required_n_per_group", "daily_traffic",
                    "treatment_split", "days_required", "weeks_required"]:
            assert key in result

    def test_days_positive(self):
        result = runtime_estimate(1000, 500)
        assert result["days_required"] > 0

    def test_more_traffic_fewer_days(self):
        r_low  = runtime_estimate(5000, 1000)
        r_high = runtime_estimate(5000, 5000)
        assert r_high["days_required"] < r_low["days_required"]

    def test_weeks_consistent_with_days(self):
        result = runtime_estimate(1000, 500)
        assert abs(result["weeks_required"] - result["days_required"] / 7) < 0.01

    def test_split_affects_days(self):
        """A 25% split means fewer users per group per day → more days."""
        r_50pct = runtime_estimate(5000, 2000, treatment_split=0.50)
        r_25pct = runtime_estimate(5000, 2000, treatment_split=0.25)
        assert r_25pct["days_required"] > r_50pct["days_required"]

    def test_formula_correct(self):
        n, daily, split = 3000, 2000, 0.5
        result = runtime_estimate(n, daily, split)
        expected_days = n / (daily * split)
        assert abs(result["days_required"] - expected_days) < 0.01


# ─── auto_power_analysis ─────────────────────────────────────────────────────

class TestAutoPowerAnalysis:

    def test_returns_dict(self, user_df):
        result = auto_power_analysis(user_df, outcome="avg_session_dur")
        assert isinstance(result, dict)

    def test_expected_keys(self, user_df):
        result = auto_power_analysis(user_df, outcome="avg_session_dur")
        for key in ["baseline_mean", "baseline_std", "n_observed",
                    "ttest_requirement", "achieved_power",
                    "mde_at_n", "sensitivity_table"]:
            assert key in result

    def test_ancova_keys_present_with_r_squared(self, user_df):
        result = auto_power_analysis(user_df, outcome="avg_session_dur",
                                     r_squared=0.25)
        assert "ancova_requirement" in result
        assert "sample_size_reduction" in result

    def test_baseline_mean_from_control(self, user_df):
        result = auto_power_analysis(user_df, outcome="avg_session_dur")
        ctrl_mean = user_df.loc[user_df["variant"] == "control",
                                "avg_session_dur"].mean()
        assert abs(result["baseline_mean"] - ctrl_mean) < 0.01

    def test_achieved_power_between_0_and_1(self, user_df):
        result = auto_power_analysis(user_df, outcome="avg_session_dur")
        assert 0 <= result["achieved_power"] <= 1

    def test_sensitivity_table_is_dataframe(self, user_df):
        result = auto_power_analysis(user_df, outcome="avg_session_dur")
        assert isinstance(result["sensitivity_table"], pd.DataFrame)

    def test_sample_size_reduction_positive(self, user_df):
        result = auto_power_analysis(user_df, outcome="avg_session_dur",
                                     r_squared=0.25)
        assert result["sample_size_reduction"] > 0

    def test_ttest_requirement_is_power_result(self, user_df):
        result = auto_power_analysis(user_df, outcome="avg_session_dur")
        assert isinstance(result["ttest_requirement"], PowerResult)

    def test_ancova_smaller_than_ttest(self, user_df):
        result = auto_power_analysis(user_df, outcome="avg_session_dur",
                                     r_squared=0.30)
        assert (result["ancova_requirement"].n_per_group <
                result["ttest_requirement"].n_per_group)
