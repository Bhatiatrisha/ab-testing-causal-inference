# tests/conftest.py
#
# Shared pytest configuration and fixtures.
# Fixtures defined here are available to all test files automatically.

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Make src/ importable from any test without needing to install the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ─── SHARED CONSTANTS ─────────────────────────────────────────────────────────

BASELINE_MEAN = 200.0
BASELINE_STD  = 80.0
N_USERS       = 500       # small enough for fast tests
RANDOM_SEED   = 42


# ─── SHARED FIXTURES ──────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def rng():
    """Session-scoped RNG — consistent across all tests in a run."""
    return np.random.default_rng(RANDOM_SEED)


@pytest.fixture(scope="session")
def base_user_df():
    """
    Session-scoped user DataFrame — generated once and reused.
    500 users, balanced split, +15% treatment effect, two covariates.
    Suitable as input to causal_models, stats_tests, and power_analysis.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n   = N_USERS
    tenure  = rng.exponential(scale=60, size=n).clip(1, 365)
    pages   = rng.poisson(lam=3, size=n).clip(1, 10).astype(float)
    variant = np.where(np.arange(n) < n // 2, "control", "treatment")
    base    = BASELINE_MEAN + tenure * 0.5 + pages * 10
    dur     = base * (1 + 0.15 * (variant == "treatment")) + rng.normal(0, 20, n)

    return pd.DataFrame({
        "user_id":              [f"u{i}" for i in range(n)],
        "variant":              variant,
        "tenure_days":          tenure,
        "avg_pages_per_sess":   pages,
        "avg_events_per_sess":  pages * 2.1 + rng.normal(0, 0.5, n),
        "avg_session_dur":      dur.clip(10),
        "log_avg_session_dur":  np.log1p(dur.clip(10)),
        "pre_avg_session_dur":  (dur / 1.15 + rng.normal(0, 15, n)).clip(10),
    })


@pytest.fixture(scope="session")
def ctrl_series(base_user_df):
    return base_user_df.loc[base_user_df["variant"] == "control",
                            "avg_session_dur"].reset_index(drop=True)


@pytest.fixture(scope="session")
def trt_series(base_user_df):
    return base_user_df.loc[base_user_df["variant"] == "treatment",
                            "avg_session_dur"].reset_index(drop=True)
