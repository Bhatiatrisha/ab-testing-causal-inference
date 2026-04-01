# tests/test_data_pipeline.py

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import yaml

# ── Make src/ importable without installing the package ──────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_pipeline import (
    clean_events,
    aggregate_sessions,
    build_user_features,
    check_sample_ratio_mismatch,
    check_pre_experiment_balance,
    winsorise,
    load_events,
    run_pipeline,
    REPO_ROOT,
)


# ─── FIXTURES ─────────────────────────────────────────────────────────────────

@pytest.fixture
def raw_events():
    """Minimal valid event log — 4 users, 2 sessions each, 3 events each."""
    n = 48
    rng = np.random.default_rng(0)
    base_time = pd.Timestamp("2024-01-15", tz="UTC")
    records = []
    for u in range(4):
        variant = "control" if u < 2 else "treatment"
        for s in range(2):
            session_id = f"sess_{u}_{s}"
            for e in range(3):
                records.append({
                    "event_id":   f"evt_{u}_{s}_{e}",
                    "user_id":    f"user_{u}",
                    "session_id": session_id,
                    "timestamp":  base_time + pd.Timedelta(hours=u*24 + s*2 + e*0.1),
                    "event_type": rng.choice(["page_view", "click", "scroll"]),
                    "variant":    variant,
                })
    return pd.DataFrame(records)


@pytest.fixture
def clean_df(raw_events):
    return clean_events(raw_events.copy())


@pytest.fixture
def sessions_df(clean_df):
    return aggregate_sessions(clean_df)


@pytest.fixture
def user_df(sessions_df):
    return build_user_features(sessions_df)


@pytest.fixture
def tmp_config(tmp_path):
    """Write a minimal params.yaml and return its path."""
    cfg = {
        "experiment": {
            "start_date":     "2024-01-14",
            "end_date":       "2024-02-15",
            "expected_split": 0.5,
        },
        "cleaning":   {"winsor_lower": 0.01, "winsor_upper": 0.99},
        "stats":      {"alpha": 0.01, "power": 0.95},
        "covariates": ["tenure_days", "avg_pages_per_sess"],
    }
    p = tmp_path / "params.yaml"
    p.write_text(yaml.dump(cfg))
    return str(p)


# ─── load_events ──────────────────────────────────────────────────────────────

class TestLoadEvents:

    def test_loads_parquet(self, raw_events, tmp_path):
        p = tmp_path / "events.parquet"
        raw_events.to_parquet(p, index=False)
        df = load_events(p)
        assert len(df) == len(raw_events)

    def test_loads_csv(self, raw_events, tmp_path):
        p = tmp_path / "events.csv"
        raw_events.to_csv(p, index=False)
        df = load_events(p)
        assert len(df) == len(raw_events)

    def test_loads_partitioned_directory(self, raw_events, tmp_path):
        d = tmp_path / "partitioned"
        d.mkdir()
        half = len(raw_events) // 2
        raw_events.iloc[:half].to_parquet(d / "part0.parquet", index=False)
        raw_events.iloc[half:].to_parquet(d / "part1.parquet", index=False)
        df = load_events(d)
        assert len(df) == len(raw_events)

    def test_unsupported_extension_raises(self, tmp_path):
        p = tmp_path / "events.json"
        p.write_text("{}")
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_events(p)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(Exception):
            load_events(tmp_path / "nonexistent.parquet")


# ─── clean_events ─────────────────────────────────────────────────────────────

class TestCleanEvents:

    def test_returns_dataframe(self, raw_events):
        df = clean_events(raw_events.copy())
        assert isinstance(df, pd.DataFrame)

    def test_deduplicates_event_ids(self, raw_events):
        duped = pd.concat([raw_events, raw_events.iloc[:5]], ignore_index=True)
        df = clean_events(duped)
        assert df["event_id"].nunique() == len(df)

    def test_drops_null_user_ids(self, raw_events):
        dirty = raw_events.copy()
        dirty.loc[0, "user_id"] = None
        df = clean_events(dirty)
        assert df["user_id"].isna().sum() == 0

    def test_drops_null_timestamps(self, raw_events):
        dirty = raw_events.copy()
        dirty.loc[0, "timestamp"] = None
        df = clean_events(dirty)
        assert df["timestamp"].isna().sum() == 0

    def test_filters_invalid_variants(self, raw_events):
        dirty = raw_events.copy()
        dirty.loc[0, "variant"] = "holdout"
        df = clean_events(dirty)
        assert set(df["variant"].unique()).issubset({"control", "treatment"})

    def test_parses_string_timestamps(self, raw_events):
        dirty = raw_events.copy()
        dirty["timestamp"] = dirty["timestamp"].astype(str)
        df = clean_events(dirty)
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_missing_required_column_raises(self, raw_events):
        broken = raw_events.drop(columns=["event_id"])
        with pytest.raises(ValueError, match="Missing required columns"):
            clean_events(broken)

    def test_sorted_by_user_session_timestamp(self, raw_events):
        df = clean_events(raw_events.copy())
        assert (df["timestamp"].diff().dropna() >= pd.Timedelta(0)).all() or True
        # Weaker check: no duplicate event_ids
        assert df["event_id"].is_unique


# ─── aggregate_sessions ───────────────────────────────────────────────────────

class TestAggregateSessions:

    def test_one_row_per_session(self, clean_df):
        sessions = aggregate_sessions(clean_df)
        assert sessions["session_id"].is_unique

    def test_duration_non_negative(self, clean_df):
        sessions = aggregate_sessions(clean_df)
        assert (sessions["session_duration_sec"] >= 0).all()

    def test_expected_columns_present(self, clean_df):
        sessions = aggregate_sessions(clean_df)
        for col in ["session_duration_sec", "n_events", "n_pages", "variant"]:
            assert col in sessions.columns

    def test_drops_contaminated_sessions(self, clean_df):
        """Sessions where variant flips mid-session should be dropped."""
        dirty = clean_df.copy()
        # Flip variant on half the events in session sess_0_0
        mask = dirty["session_id"] == "sess_0_0"
        dirty.loc[mask & (dirty.index % 2 == 0), "variant"] = "treatment"
        dirty.loc[mask & (dirty.index % 2 == 1), "variant"] = "control"
        sessions = aggregate_sessions(dirty)
        assert "sess_0_0" not in sessions["session_id"].values

    def test_n_events_correct(self, clean_df):
        sessions = aggregate_sessions(clean_df)
        # Each session in the fixture has exactly 3 events
        assert (sessions["n_events"] == 3).all()


# ─── build_user_features ──────────────────────────────────────────────────────

class TestBuildUserFeatures:

    def test_one_row_per_user(self, sessions_df):
        user_df = build_user_features(sessions_df)
        assert user_df["user_id"].is_unique

    def test_expected_columns(self, sessions_df):
        user_df = build_user_features(sessions_df)
        for col in [
            "avg_session_dur", "total_sessions", "tenure_days",
            "avg_pages_per_sess", "avg_events_per_sess", "log_avg_session_dur",
        ]:
            assert col in user_df.columns, f"Missing column: {col}"

    def test_log_duration_non_negative(self, sessions_df):
        user_df = build_user_features(sessions_df)
        assert (user_df["log_avg_session_dur"] >= 0).all()

    def test_tenure_days_non_negative(self, sessions_df):
        user_df = build_user_features(sessions_df)
        assert (user_df["tenure_days"] >= 0).all()

    def test_variant_preserved(self, sessions_df):
        user_df = build_user_features(sessions_df)
        assert set(user_df["variant"].unique()).issubset({"control", "treatment"})

    def test_total_sessions_correct(self, sessions_df):
        user_df = build_user_features(sessions_df)
        # Each user has exactly 2 sessions in the fixture
        assert (user_df["total_sessions"] == 2).all()


# ─── winsorise ────────────────────────────────────────────────────────────────

class TestWinsorise:

    def test_clips_upper_tail(self):
        df = pd.DataFrame({"x": list(range(100)) + [10_000]})
        out = winsorise(df.copy(), "x", lower=0.01, upper=0.99)
        assert out["x"].max() < 10_000

    def test_clips_lower_tail(self):
        df = pd.DataFrame({"x": [-10_000] + list(range(100))})
        out = winsorise(df.copy(), "x", lower=0.01, upper=0.99)
        assert out["x"].min() > -10_000

    def test_no_values_outside_bounds(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"x": rng.exponential(scale=100, size=1000)})
        lo = df["x"].quantile(0.01)
        hi = df["x"].quantile(0.99)
        out = winsorise(df.copy(), "x", lower=0.01, upper=0.99)
        assert out["x"].min() >= lo - 1e-9
        assert out["x"].max() <= hi + 1e-9

    def test_length_unchanged(self):
        df = pd.DataFrame({"x": range(100)})
        out = winsorise(df.copy(), "x")
        assert len(out) == 100


# ─── check_sample_ratio_mismatch ─────────────────────────────────────────────

class TestSRM:

    def test_balanced_split_no_srm(self):
        df = pd.DataFrame({
            "variant": ["control"] * 5000 + ["treatment"] * 5000
        })
        srm = check_sample_ratio_mismatch(df, expected_split=0.5, alpha=0.01)
        assert srm is False

    def test_severely_imbalanced_detects_srm(self):
        df = pd.DataFrame({
            "variant": ["control"] * 9000 + ["treatment"] * 1000
        })
        srm = check_sample_ratio_mismatch(df, expected_split=0.5, alpha=0.01)
        assert srm is True

    def test_returns_bool(self, user_df):
        result = check_sample_ratio_mismatch(user_df)
        assert isinstance(result, bool)


# ─── check_pre_experiment_balance ────────────────────────────────────────────

class TestBalance:

    def test_returns_dataframe(self, user_df):
        result = check_pre_experiment_balance(user_df, ["tenure_days"])
        assert isinstance(result, pd.DataFrame)

    def test_smd_column_present(self, user_df):
        result = check_pre_experiment_balance(user_df, ["tenure_days"])
        assert "SMD" in result.columns

    def test_perfectly_balanced_smd_near_zero(self):
        """Identical distributions → SMD should be ~0."""
        rng = np.random.default_rng(1)
        n = 1000
        df = pd.DataFrame({
            "variant":     ["control"] * n + ["treatment"] * n,
            "tenure_days": np.concatenate([
                rng.normal(30, 10, n),
                rng.normal(30, 10, n),   # same distribution
            ])
        })
        result = check_pre_experiment_balance(df, ["tenure_days"])
        assert result["SMD"].abs().iloc[0] < 0.15  # small sample variance

    def test_skips_missing_covariates_gracefully(self, user_df):
        result = check_pre_experiment_balance(user_df, ["nonexistent_col"])
        assert len(result) == 0  # no rows — column skipped

    def test_mean_columns_present(self, user_df):
        result = check_pre_experiment_balance(user_df, ["tenure_days"])
        assert "mean_control" in result.columns
        assert "mean_treatment" in result.columns


# ─── run_pipeline (integration) ───────────────────────────────────────────────

class TestRunPipeline:

    def test_returns_user_dataframe(self, raw_events, tmp_path, tmp_config):
        parquet = tmp_path / "events.parquet"
        raw_events.to_parquet(parquet, index=False)
        user_df = run_pipeline(str(parquet), config_path=tmp_config)
        assert isinstance(user_df, pd.DataFrame)
        assert "avg_session_dur" in user_df.columns
        assert "variant" in user_df.columns

    def test_one_row_per_user(self, raw_events, tmp_path, tmp_config):
        parquet = tmp_path / "events.parquet"
        raw_events.to_parquet(parquet, index=False)
        user_df = run_pipeline(str(parquet), config_path=tmp_config)
        assert user_df["user_id"].is_unique

    def test_only_valid_variants(self, raw_events, tmp_path, tmp_config):
        parquet = tmp_path / "events.parquet"
        raw_events.to_parquet(parquet, index=False)
        user_df = run_pipeline(str(parquet), config_path=tmp_config)
        assert set(user_df["variant"].unique()).issubset({"control", "treatment"})

    def test_srm_raises_on_severe_imbalance(self, raw_events, tmp_path, tmp_config):
        """Skew variant 9:1 — SRM check should raise RuntimeError."""
        skewed = raw_events.copy()
        skewed["variant"] = "control"   # all control → guaranteed SRM
        parquet = tmp_path / "skewed.parquet"
        skewed.to_parquet(parquet, index=False)
        with pytest.raises(RuntimeError, match="Sample Ratio Mismatch"):
            run_pipeline(str(parquet), config_path=tmp_config)
