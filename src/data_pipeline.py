# src/data_pipeline.py

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import chi2_contingency
import logging
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
def load_config(path: str = "config/params.yaml") -> dict:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found at: {config_path}\n"
            f"REPO_ROOT resolved to: {REPO_ROOT}\n"
            f"Files in REPO_ROOT: {list(REPO_ROOT.iterdir())}"
        )
    with open(config_path) as f:
        return yaml.safe_load(f)


# ─── 1. LOAD ─────────────────────────────────────────────────────────────────

def load_events(source: str | Path, **kwargs) -> pd.DataFrame:
    """
    Load raw event logs from CSV, Parquet, or a directory of partitioned files.
    Supports 500K+ rows efficiently via chunked CSV or Parquet columnar reads.
    """
    source = Path(source)

    if source.is_dir():
        # Partitioned parquet (e.g. date-partitioned export from warehouse)
        dfs = [pd.read_parquet(f) for f in sorted(source.glob("*.parquet"))]
        df = pd.concat(dfs, ignore_index=True)
    elif source.suffix == ".parquet":
        df = pd.read_parquet(source, **kwargs)
    elif source.suffix in (".csv", ".gz"):
        # For very large CSVs, read in chunks to avoid OOM
        chunk_size = kwargs.pop("chunksize", 100_000)
        chunks = pd.read_csv(source, chunksize=chunk_size, **kwargs)
        df = pd.concat(chunks, ignore_index=True)
    else:
        raise ValueError(f"Unsupported file type: {source.suffix}")

    logger.info(f"Loaded {len(df):,} raw events from {source}")
    return df


# ─── 2. CLEAN & VALIDATE SCHEMA ──────────────────────────────────────────────

REQUIRED_COLS = {
    "event_id":   "object",
    "user_id":    "object",
    "session_id": "object",
    "timestamp":  "datetime64[ns]",
    "event_type": "object",
    "variant":    "object",   # 'control' or 'treatment'
}

def clean_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Enforce schema & dtypes
    - Parse timestamps
    - Drop nulls in critical columns
    - Deduplicate on event_id
    - Filter to valid variant labels
    """
    # Schema check
    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Parse timestamps if not already parsed
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    # Drop rows with null in critical columns
    critical = ["user_id", "session_id", "timestamp", "variant"]
    before = len(df)
    df = df.dropna(subset=critical)
    logger.info(f"Dropped {before - len(df):,} rows with nulls in critical columns")

    # Deduplicate events
    before = len(df)
    df = df.drop_duplicates(subset=["event_id"])
    logger.info(f"Removed {before - len(df):,} duplicate events")

    # Keep only valid variant labels
    valid_variants = {"control", "treatment"}
    before = len(df)
    df = df[df["variant"].isin(valid_variants)]
    logger.info(f"Removed {before - len(df):,} rows with invalid variant labels")

    # Sort for downstream session aggregation
    df = df.sort_values(["user_id", "session_id", "timestamp"]).reset_index(drop=True)
    logger.info(f"Clean event log: {len(df):,} events, {df['user_id'].nunique():,} users")
    return df


# ─── 3. SESSION AGGREGATION ───────────────────────────────────────────────────

def aggregate_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse event-level rows into one row per (user_id, session_id).
    Key metrics:
      - session_duration_sec: last_event - first_event in seconds
      - n_events: count of events in session
      - n_pages: distinct page/screen identifiers visited
      - variant: should be constant per session (validated below)
    """
    agg = (
        df.groupby(["user_id", "session_id"])
        .agg(
            session_start   = ("timestamp", "min"),
            session_end     = ("timestamp", "max"),
            n_events        = ("event_id",  "count"),
            n_pages         = ("event_type","nunique"),
            variant         = ("variant",   "first"),   # enforced consistent below
        )
        .reset_index()
    )

    # Duration in seconds
    agg["session_duration_sec"] = (
        (agg["session_end"] - agg["session_start"])
        .dt.total_seconds()
        .clip(lower=0)
    )

    # Detect sessions where variant flipped mid-session (data quality issue)
    variant_counts = df.groupby("session_id")["variant"].nunique()
    contaminated = variant_counts[variant_counts > 1].index
    if len(contaminated):
        logger.warning(f"{len(contaminated):,} sessions had multiple variants — dropping")
        agg = agg[~agg["session_id"].isin(contaminated)]

    logger.info(f"Aggregated to {len(agg):,} sessions, {agg['user_id'].nunique():,} users")
    return agg


# ─── 4. USER-LEVEL FEATURES ───────────────────────────────────────────────────

def build_user_features(sessions: pd.DataFrame) -> pd.DataFrame:
    """
    Roll up sessions to user level — one row per user.
    Used as the unit of analysis for DiD / PSM.
    """
    user_df = (
        sessions.groupby(["user_id", "variant"])
        .agg(
            total_sessions      = ("session_id",          "count"),
            avg_session_dur     = ("session_duration_sec","mean"),
            median_session_dur  = ("session_duration_sec","median"),
            total_session_dur   = ("session_duration_sec","sum"),
            avg_pages_per_sess  = ("n_pages",             "mean"),
            avg_events_per_sess = ("n_events",            "mean"),
            first_seen          = ("session_start",       "min"),
            last_seen           = ("session_end",         "max"),
        )
        .reset_index()
    )

    # Tenure (days between first and last activity)
    user_df["tenure_days"] = (
        (user_df["last_seen"] - user_df["first_seen"])
        .dt.total_seconds() / 86_400
    )

    # Log-transform session duration to reduce right-skew (common for session data)
    user_df["log_avg_session_dur"] = np.log1p(user_df["avg_session_dur"])

    logger.info(f"Built user-level features: {len(user_df):,} users")
    return user_df


# ─── 5. VALIDATION CHECKS ─────────────────────────────────────────────────────

def check_sample_ratio_mismatch(user_df: pd.DataFrame, expected_split: float = 0.5,
                                 alpha: float = 0.01) -> bool:
    """
    Sample Ratio Mismatch (SRM) check using chi-square test.
    If traffic allocation is uneven (p < alpha), the experiment is compromised.
    Returns True if SRM is detected (you should STOP and investigate).
    """
    counts = user_df["variant"].value_counts()
    n_total = counts.sum()
    expected = [n_total * expected_split, n_total * (1 - expected_split)]
    observed = [counts.get("control", 0), counts.get("treatment", 0)]

    chi2, p_val, *_ = chi2_contingency([observed, expected])
    srm_detected = bool(p_val < alpha)

    logger.info(
        f"SRM check — control: {observed[0]:,}, treatment: {observed[1]:,}, "
        f"chi2={chi2:.3f}, p={p_val:.4f} → {'SRM DETECTED' if srm_detected else 'OK'}"
    )
    return srm_detected


def winsorise(df: pd.DataFrame, col: str, lower: float = 0.01,
               upper: float = 0.99) -> pd.DataFrame:
    """Cap extreme values at the specified quantile bounds."""
    lo = df[col].quantile(lower)
    hi = df[col].quantile(upper)
    df[col] = df[col].clip(lo, hi)
    logger.info(f"Winsorised {col} to [{lo:.1f}, {hi:.1f}]")
    return df


def check_pre_experiment_balance(user_df: pd.DataFrame,
                                  covariates: list[str]) -> pd.DataFrame:
    """
    Compute Standardised Mean Difference (SMD) for each covariate.
    SMD < 0.1 is considered well-balanced.
    Returns a summary DataFrame for reporting.
    """
    ctrl = user_df[user_df["variant"] == "control"]
    trt  = user_df[user_df["variant"] == "treatment"]
    rows = []
    for col in covariates:
        if col not in user_df.columns:
            continue
        mean_c, std_c = ctrl[col].mean(), ctrl[col].std()
        mean_t, std_t = trt[col].mean(),  trt[col].std()
        pooled_std = np.sqrt((std_c**2 + std_t**2) / 2)
        smd = (mean_t - mean_c) / pooled_std if pooled_std > 0 else np.nan
        rows.append({"covariate": col, "mean_control": mean_c,
                     "mean_treatment": mean_t, "SMD": smd})
    balance_df = pd.DataFrame(rows)
    imbalanced = balance_df[balance_df["SMD"].abs() > 0.1]
    if len(imbalanced):
        logger.warning(f"Imbalanced covariates (SMD > 0.1):\n{imbalanced}")
    return balance_df


# ─── 6. MAIN PIPELINE ─────────────────────────────────────────────────────────

def run_pipeline(source: str,
                 config_path: str = "config/params.yaml") -> pd.DataFrame:
    cfg = load_config(config_path)

    # Resolve source relative to repo root, not cwd
    source_path = Path(source)
    if not source_path.is_absolute():
        source_path = REPO_ROOT / source_path
    if not source_path.exists():
        raise FileNotFoundError(
            f"Event data not found at: {source_path}\n"
            f"Run src/simulation.py first to generate it.\n"
            f"Files in data/raw: {list((REPO_ROOT / 'data' / 'raw').iterdir()) if (REPO_ROOT / 'data' / 'raw').exists() else 'directory missing'}"
        )

    df = load_events(source_path)

    # 2. Clean
    df = clean_events(df)

    # 3. Filter to experiment window
    start = pd.Timestamp(cfg["experiment"]["start_date"], tz="UTC")
    end   = pd.Timestamp(cfg["experiment"]["end_date"],   tz="UTC")
    df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
    logger.info(f"Filtered to experiment window: {len(df):,} events remain")

    # 4. Session aggregation
    sessions = aggregate_sessions(df)

    # 5. Winsorise session duration before rolling up
    sessions = winsorise(sessions, "session_duration_sec",
                          lower=cfg["cleaning"]["winsor_lower"],
                          upper=cfg["cleaning"]["winsor_upper"])

    # 6. User-level features
    user_df = build_user_features(sessions)

    # 7. SRM check — halt if mismatch detected
    srm = check_sample_ratio_mismatch(
        user_df,
        expected_split=cfg["experiment"]["expected_split"],
        alpha=cfg["stats"]["alpha"]
    )
    if srm:
        raise RuntimeError("Sample Ratio Mismatch detected. Investigate before proceeding.")

    # 8. Pre-experiment balance check
    covariates = cfg.get("covariates", ["tenure_days", "avg_pages_per_sess"])
    balance_df = check_pre_experiment_balance(user_df, covariates)
    logger.info(f"Balance summary:\n{balance_df.to_string(index=False)}")

    return user_df


if __name__ == "__main__":
    if __name__ == "__main__":
        import sys

    # Detect Jupyter/IPython — argv is not reliable there
    try:
        get_ipython()
        in_notebook = True
    except NameError:
        in_notebook = False

    if in_notebook:
        source = "data/raw/events.parquet"
    else:
        source = sys.argv[1] if len(sys.argv) > 1 else "data/raw/events.parquet"

    result = run_pipeline(source)
    output = Path("data/processed/user_features.parquet")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    logger.info(f"Saved analysis-ready data to {output}")
