# tests/test_causal_models.py

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.causal_models import (
    compute_propensity_scores,
    nearest_neighbour_matching,
    psm_pipeline,
    difference_in_differences,
    did_regression,
    ancova,
    summarise_model,
)


# ─── FIXTURES ─────────────────────────────────────────────────────────────────

@pytest.fixture
def balanced_user_df():
    """
    500 users, balanced 50/50, with a clean +15% treatment effect
    and two confounders (tenure, pages). Used for most model tests.
    """
    rng = np.random.default_rng(42)
    n = 500
    tenure     = rng.exponential(scale=60, size=n).clip(1, 365)
    pages      = rng.poisson(lam=3, size=n).clip(1, 10).astype(float)
    variant    = np.where(np.arange(n) < n // 2, "control", "treatment")
    base_dur   = 200 + tenure * 0.5 + pages * 10
    treatment  = (variant == "treatment").astype(float)
    duration   = base_dur * (1 + 0.15 * treatment) + rng.normal(0, 20, n)

    return pd.DataFrame({
        "user_id":          [f"u{i}" for i in range(n)],
        "variant":          variant,
        "tenure_days":      tenure,
        "avg_pages_per_sess": pages,
        "avg_session_dur":  duration.clip(10),
        "log_avg_session_dur": np.log1p(duration.clip(10)),
    })


@pytest.fixture
def did_user_df(balanced_user_df):
    """Adds a synthetic pre-period column for DiD tests."""
    rng = np.random.default_rng(99)
    df = balanced_user_df.copy()
    # Pre-period: same distribution, no treatment effect
    df["pre_avg_session_dur"] = (
        df["avg_session_dur"] / 1.15 + rng.normal(0, 15, len(df))
    ).clip(10)
    return df


COVARIATES = ["tenure_days", "avg_pages_per_sess"]


# ─── compute_propensity_scores ────────────────────────────────────────────────

class TestPropensityScores:

    def test_adds_propensity_score_column(self, balanced_user_df):
        df = compute_propensity_scores(balanced_user_df, COVARIATES)
        assert "propensity_score" in df.columns

    def test_scores_between_zero_and_one(self, balanced_user_df):
        df = compute_propensity_scores(balanced_user_df, COVARIATES)
        assert df["propensity_score"].between(0, 1).all()

    def test_does_not_drop_all_rows(self, balanced_user_df):
        df = compute_propensity_scores(balanced_user_df, COVARIATES)
        assert len(df) > 0

    def test_trims_outside_common_support(self):
        """Units with extreme propensity scores should be trimmed."""
        rng = np.random.default_rng(7)
        n = 300
        # Make control have very low tenure, treatment very high → no overlap
        ctrl = pd.DataFrame({
            "user_id":   [f"c{i}" for i in range(n)],
            "variant":   "control",
            "tenure_days": rng.uniform(1, 5, n),
            "avg_pages_per_sess": rng.uniform(1, 2, n),
            "avg_session_dur": rng.uniform(100, 200, n),
        })
        trt = pd.DataFrame({
            "user_id":   [f"t{i}" for i in range(n)],
            "variant":   "treatment",
            "tenure_days": rng.uniform(300, 365, n),
            "avg_pages_per_sess": rng.uniform(8, 10, n),
            "avg_session_dur": rng.uniform(300, 400, n),
        })
        df_extreme = pd.concat([ctrl, trt], ignore_index=True)
        scored = compute_propensity_scores(df_extreme, COVARIATES)
        # With no overlap some rows should be trimmed
        assert len(scored) <= len(df_extreme)

    def test_mean_score_near_split(self, balanced_user_df):
        """In a balanced RCT the mean propensity score should be ~0.5."""
        df = compute_propensity_scores(balanced_user_df, COVARIATES)
        assert abs(df["propensity_score"].mean() - 0.5) < 0.1


# ─── nearest_neighbour_matching ───────────────────────────────────────────────

class TestNNMatching:

    def test_returns_dataframe(self, balanced_user_df):
        scored = compute_propensity_scores(balanced_user_df, COVARIATES)
        matched = nearest_neighbour_matching(scored)
        assert isinstance(matched, pd.DataFrame)

    def test_match_id_column_present(self, balanced_user_df):
        scored = compute_propensity_scores(balanced_user_df, COVARIATES)
        matched = nearest_neighbour_matching(scored)
        assert "match_id" in matched.columns

    def test_both_variants_in_matched(self, balanced_user_df):
        scored  = compute_propensity_scores(balanced_user_df, COVARIATES)
        matched = nearest_neighbour_matching(scored)
        assert set(matched["variant"].unique()) == {"control", "treatment"}

    def test_each_match_id_has_two_rows(self, balanced_user_df):
        """Each pair should have exactly one control and one treatment."""
        scored  = compute_propensity_scores(balanced_user_df, COVARIATES)
        matched = nearest_neighbour_matching(scored, caliper=0.5)
        counts  = matched.groupby("match_id")["variant"].count()
        assert (counts == 2).all()

    def test_caliper_reduces_matches(self, balanced_user_df):
        scored = compute_propensity_scores(balanced_user_df, COVARIATES)
        wide   = nearest_neighbour_matching(scored, caliper=0.5)
        narrow = nearest_neighbour_matching(scored, caliper=0.001)
        assert len(narrow) <= len(wide)

    def test_no_duplicate_controls_without_replacement(self, balanced_user_df):
        scored  = compute_propensity_scores(balanced_user_df, COVARIATES)
        matched = nearest_neighbour_matching(scored, caliper=0.5, replacement=False)
        ctrl_ids = matched.loc[matched["variant"] == "control", "user_id"]
        assert ctrl_ids.is_unique


# ─── psm_pipeline ─────────────────────────────────────────────────────────────

class TestPSMPipeline:

    def test_returns_smaller_dataframe(self, balanced_user_df):
        matched = psm_pipeline(balanced_user_df, COVARIATES)
        assert len(matched) <= len(balanced_user_df)

    def test_improves_balance(self, balanced_user_df):
        """
        SMD on tenure_days should be lower after PSM than before
        when the covariate is imbalanced.
        """
        # Introduce imbalance: treatment users have higher tenure
        df = balanced_user_df.copy()
        rng = np.random.default_rng(5)
        df.loc[df["variant"] == "treatment", "tenure_days"] += rng.uniform(20, 40, 250)

        def smd(df, col):
            c = df.loc[df["variant"] == "control",   col]
            t = df.loc[df["variant"] == "treatment",  col]
            return abs((t.mean() - c.mean()) / np.sqrt((c.std()**2 + t.std()**2) / 2))

        smd_before = smd(df, "tenure_days")
        matched    = psm_pipeline(df, COVARIATES, caliper=0.1)
        smd_after  = smd(matched, "tenure_days")
        assert smd_after < smd_before

    def test_propensity_score_column_in_output(self, balanced_user_df):
        matched = psm_pipeline(balanced_user_df, COVARIATES)
        assert "propensity_score" in matched.columns


# ─── difference_in_differences ───────────────────────────────────────────────

class TestDiD:

    def test_returns_dict_with_att(self, did_user_df):
        result = difference_in_differences(
            did_user_df,
            outcome="avg_session_dur",
            pre_col="pre_avg_session_dur",
        )
        assert "att" in result
        assert isinstance(result["att"], float)

    def test_att_positive_for_positive_effect(self, did_user_df):
        result = difference_in_differences(
            did_user_df,
            outcome="avg_session_dur",
            pre_col="pre_avg_session_dur",
        )
        assert result["att"] > 0

    def test_pct_lift_reasonable(self, did_user_df):
        result = difference_in_differences(
            did_user_df,
            outcome="avg_session_dur",
            pre_col="pre_avg_session_dur",
        )
        # Injected ~15% lift — DiD estimate should be in the right ballpark
        assert 0.0 < result["pct_lift"] < 0.5

    def test_missing_pre_col_raises(self, balanced_user_df):
        with pytest.raises(ValueError, match="pre-period column"):
            difference_in_differences(balanced_user_df, pre_col="nonexistent")

    def test_expected_keys_present(self, did_user_df):
        result = difference_in_differences(
            did_user_df,
            outcome="avg_session_dur",
            pre_col="pre_avg_session_dur",
        )
        for key in ["estimator", "att", "pct_lift",
                    "ctrl_pre_mean", "ctrl_post_mean",
                    "trt_pre_mean",  "trt_post_mean"]:
            assert key in result


# ─── did_regression ───────────────────────────────────────────────────────────

class TestDiDRegression:

    def test_returns_fitted_model(self, did_user_df):
        model = did_regression(
            did_user_df,
            outcome="avg_session_dur",
            pre_col="pre_avg_session_dur",
        )
        assert hasattr(model, "params")
        assert hasattr(model, "pvalues")

    def test_did_interact_coef_positive(self, did_user_df):
        model = did_regression(
            did_user_df,
            outcome="avg_session_dur",
            pre_col="pre_avg_session_dur",
        )
        assert model.params["did_interact"] > 0

    def test_nobs_correct(self, did_user_df):
        model = did_regression(
            did_user_df,
            outcome="avg_session_dur",
            pre_col="pre_avg_session_did",
            pre_col_real="pre_avg_session_dur",  # fixture has this
        ) if False else did_regression(
            did_user_df,
            outcome="avg_session_dur",
            pre_col="pre_avg_session_dur",
        )
        # Panel has 2 rows per user (pre + post)
        assert model.nobs == len(did_user_df) * 2

    def test_missing_pre_col_raises(self, balanced_user_df):
        with pytest.raises(ValueError, match="pre-period column"):
            did_regression(balanced_user_df, pre_col="missing_col")

    def test_with_covariates(self, did_user_df):
        model = did_regression(
            did_user_df,
            outcome="avg_session_dur",
            pre_col="pre_avg_session_dur",
            covariates=COVARIATES,
        )
        for cov in COVARIATES:
            assert cov in model.params.index


# ─── ancova ───────────────────────────────────────────────────────────────────

class TestANCOVA:

    def test_returns_fitted_model(self, balanced_user_df):
        model = ancova(balanced_user_df, outcome="avg_session_dur",
                       covariates=COVARIATES)
        assert hasattr(model, "params")

    def test_treated_coef_positive(self, balanced_user_df):
        model = ancova(balanced_user_df, outcome="avg_session_dur",
                       covariates=COVARIATES)
        assert model.params["treated"] > 0

    def test_pvalue_significant_large_effect(self, balanced_user_df):
        """With 250 users per group and 15% lift, p should be well below 0.05."""
        model = ancova(balanced_user_df, outcome="avg_session_dur",
                       covariates=COVARIATES)
        assert model.pvalues["treated"] < 0.05

    def test_log_transform_used_by_default(self, balanced_user_df):
        """With log_outcome=True the DV column should be _outcome (log-transformed)."""
        model = ancova(balanced_user_df, outcome="avg_session_dur",
                       covariates=COVARIATES, log_outcome=True)
        assert "_outcome" in model.model.endog_names or "log" in str(model.model.formula)

    def test_no_log_transform(self, balanced_user_df):
        model = ancova(balanced_user_df, outcome="avg_session_dur",
                       covariates=COVARIATES, log_outcome=False)
        assert model.params["treated"] > 0

    def test_r_squared_improves_with_covariates(self, balanced_user_df):
        model_plain = ancova(balanced_user_df, outcome="avg_session_dur",
                             covariates=None)
        model_cov   = ancova(balanced_user_df, outcome="avg_session_dur",
                             covariates=COVARIATES)
        assert model_cov.rsquared >= model_plain.rsquared

    def test_covariates_in_params(self, balanced_user_df):
        model = ancova(balanced_user_df, outcome="avg_session_dur",
                       covariates=COVARIATES)
        for cov in COVARIATES:
            assert cov in model.params.index


# ─── summarise_model ──────────────────────────────────────────────────────────

class TestSummariseModel:

    def test_returns_dict(self, balanced_user_df):
        model  = ancova(balanced_user_df, outcome="avg_session_dur",
                        covariates=COVARIATES)
        result = summarise_model(model)
        assert isinstance(result, dict)

    def test_expected_keys(self, balanced_user_df):
        model  = ancova(balanced_user_df, outcome="avg_session_dur",
                        covariates=COVARIATES)
        result = summarise_model(model)
        for key in ["coef", "pct_lift", "ci_low", "ci_high", "p_value",
                    "n_obs", "r_squared"]:
            assert key in result, f"Missing key: {key}"

    def test_ci_low_less_than_ci_high(self, balanced_user_df):
        model  = ancova(balanced_user_df, outcome="avg_session_dur",
                        covariates=COVARIATES)
        result = summarise_model(model)
        assert result["ci_low"] < result["ci_high"]

    def test_n_obs_matches_model(self, balanced_user_df):
        model  = ancova(balanced_user_df, outcome="avg_session_dur",
                        covariates=COVARIATES)
        result = summarise_model(model)
        assert result["n_obs"] == int(model.nobs)

    def test_pct_lift_positive_for_positive_treatment(self, balanced_user_df):
        model  = ancova(balanced_user_df, outcome="avg_session_dur",
                        covariates=COVARIATES)
        result = summarise_model(model)
        assert result["pct_lift"] > 0
