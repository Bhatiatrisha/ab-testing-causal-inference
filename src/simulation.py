# src/simulation.py

import numpy as np
import pandas as pd
from pathlib import Path
import uuid
import yaml
import logging
from datetime import timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─── 1. CONFIG ────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "n_users":          50_000,
    "avg_sessions":     10,          # avg sessions per user
    "treatment_split":  0.5,
    "experiment": {
        "start_date":   "2024-01-15",
        "end_date":     "2024-02-15",
    },
    "treatment_effect": {
        "session_duration_lift": 0.12,   # +12% lift to reproduce
        "affected_share":        1.0,    # fraction of treatment users affected
    },
    "confounders": {
        "enabled": True,
        # Power users get more sessions AND longer durations — classic confounder
        "power_user_share":        0.15,
        "power_user_session_mult": 2.5,
        "power_user_duration_mult":1.8,
        # Device type affects session duration independently of treatment
        "mobile_share":            0.55,
        "mobile_duration_mult":    0.70,
    },
    "session_duration": {
        "base_shape":  2.0,    # Weibull shape — controls skew
        "base_scale":  180.0,  # seconds (~3 min median baseline)
        "min_sec":     5,
        "max_sec":     3600,
    },
    "noise": {
        "duration_cv": 0.25,   # coefficient of variation for per-user noise
    },
    "random_seed": 42,
}


def load_sim_config(path: str = None) -> dict:
    if path and Path(path).exists():
        with open(path) as f:
            cfg = yaml.safe_load(f)
        # Deep merge with defaults
        merged = DEFAULT_CONFIG.copy()
        for k, v in cfg.items():
            if isinstance(v, dict) and k in merged:
                merged[k].update(v)
            else:
                merged[k] = v
        return merged
    return DEFAULT_CONFIG.copy()


# ─── 2. USER POPULATION ───────────────────────────────────────────────────────

def generate_users(cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    """
    Create a user population with:
      - variant assignment (control / treatment)
      - power-user flag (confounder correlated with both assignment and outcome)
      - device type (mobile / desktop — affects duration, not assignment)
      - tenure (days active before experiment — used as covariate in PSM)
    """
    n = cfg["n_users"]
    conf = cfg["confounders"]

    users = pd.DataFrame({"user_id": [str(uuid.uuid4()) for _ in range(n)]})

    # Variant assignment — Bernoulli, independent of confounders (clean RCT)
    users["variant"] = np.where(
        rng.random(n) < cfg["treatment_split"], "treatment", "control"
    )

    # Power-user flag — independent of variant (correct RCT), but affects outcome
    users["is_power_user"] = rng.random(n) < conf["power_user_share"]

    # Device — independent of variant
    users["device"] = np.where(rng.random(n) < conf["mobile_share"], "mobile", "desktop")

    # Tenure: right-skewed (most users are newer, some are long-time)
    users["tenure_days"] = rng.exponential(scale=60, size=n).clip(1, 730).astype(int)

    # Region (additional covariate for PSM demo)
    users["region"] = rng.choice(
        ["north", "south", "east", "west"],
        size=n, p=[0.30, 0.25, 0.25, 0.20]
    )

    logger.info(
        f"Generated {n:,} users — "
        f"treatment: {users['variant'].eq('treatment').mean():.1%}, "
        f"power users: {users['is_power_user'].mean():.1%}, "
        f"mobile: {users['device'].eq('mobile').mean():.1%}"
    )
    return users


# ─── 3. SESSION COUNT PER USER ────────────────────────────────────────────────

def assign_session_counts(users: pd.DataFrame, cfg: dict,
                           rng: np.random.Generator) -> pd.Series:
    """
    Negative-binomial session counts — overdispersed like real engagement data.
    Power users get more sessions (confounder path 1).
    """
    n = len(users)
    avg = cfg["avg_sessions"]
    conf = cfg["confounders"]

    # Base: negative binomial (mean=avg, dispersion r=5)
    r = 5.0
    p = r / (r + avg)
    base_counts = rng.negative_binomial(r, p, size=n).clip(1, 60)

    if conf["enabled"]:
        mult = np.where(users["is_power_user"], conf["power_user_session_mult"], 1.0)
        counts = (base_counts * mult).astype(int).clip(1, 100)
    else:
        counts = base_counts

    return counts


# ─── 4. SESSION TIMESTAMPS ────────────────────────────────────────────────────

def sample_timestamps(n_sessions: int, start: pd.Timestamp,
                       end: pd.Timestamp, rng: np.random.Generator) -> np.ndarray:
    """
    Sample session start times uniformly within the experiment window.
    Real traffic has day-of-week rhythm — added as a subtle sine wave.
    """
    window_sec = int((end - start).total_seconds())
    offsets = rng.integers(0, window_sec, size=n_sessions)

    # Day-of-week effect: ~20% more sessions on weekdays
    dow_bias = 0.2 * np.sin(2 * np.pi * offsets / 86_400)
    keep_prob = 0.5 + dow_bias
    # Re-sample sessions that fall on low-traffic hours (simplified thinning)
    mask = rng.random(n_sessions) < keep_prob
    offsets[~mask] = rng.integers(0, window_sec, size=(~mask).sum())

    timestamps = np.array([start + timedelta(seconds=int(s)) for s in offsets])
    return timestamps


# ─── 5. SESSION DURATION ──────────────────────────────────────────────────────

def sample_durations(n: int, base_scale: float, shape: float,
                     rng: np.random.Generator) -> np.ndarray:
    """
    Weibull-distributed durations — captures the heavy right tail of real
    session data better than log-normal alone.
    """
    return rng.weibull(shape, size=n) * base_scale


def apply_duration_multipliers(durations: np.ndarray, user_row: pd.Series,
                                 cfg: dict, rng: np.random.Generator) -> np.ndarray:
    """
    Apply per-user multipliers encoding confounders and treatment effect.
    Each multiplier is a separate causal path — makes the DGP explicit.
    """
    conf = cfg["confounders"]
    te   = cfg["treatment_effect"]
    sd   = cfg["session_duration"]

    mult = np.ones(len(durations))

    # Confounder path 1: power users have longer sessions
    if conf["enabled"] and user_row["is_power_user"]:
        mult *= conf["power_user_duration_mult"]

    # Confounder path 2: mobile users have shorter sessions
    if conf["enabled"] and user_row["device"] == "mobile":
        mult *= conf["mobile_duration_mult"]

    # Treatment effect: additive lift on the multiplier
    if user_row["variant"] == "treatment":
        affected = rng.random(len(durations)) < te["affected_share"]
        mult = np.where(affected, mult * (1 + te["session_duration_lift"]), mult)

    # Per-user random noise (heterogeneous treatment effects)
    noise = rng.normal(1.0, cfg["noise"]["duration_cv"], size=len(durations))
    mult *= np.clip(noise, 0.3, 3.0)

    return np.clip(durations * mult, sd["min_sec"], sd["max_sec"])


# ─── 6. EVENT GENERATION ──────────────────────────────────────────────────────

EVENT_TYPES = [
    "page_view", "click", "scroll", "search",
    "add_to_cart", "checkout_start", "purchase", "video_play",
]
EVENT_PROBS = [0.40, 0.25, 0.15, 0.08, 0.05, 0.03, 0.02, 0.02]


def generate_events_for_session(user_id: str, session_id: str, variant: str,
                                  session_start: pd.Timestamp,
                                  session_dur_sec: float,
                                  rng: np.random.Generator) -> list[dict]:
    """
    Scatter N events uniformly within the session window.
    N ~ Poisson(lambda proportional to session duration).
    """
    lam = max(1, session_dur_sec / 30)   # ~1 event per 30 sec
    n_events = int(rng.poisson(lam))
    if n_events == 0:
        n_events = 1

    offsets_sec = np.sort(rng.uniform(0, session_dur_sec, size=n_events))
    event_types = rng.choice(EVENT_TYPES, size=n_events, p=EVENT_PROBS)

    return [
        {
            "event_id":   str(uuid.uuid4()),
            "user_id":    user_id,
            "session_id": session_id,
            "timestamp":  session_start + timedelta(seconds=float(o)),
            "event_type": et,
            "variant":    variant,
        }
        for o, et in zip(offsets_sec, event_types)
    ]


# ─── 7. MAIN SIMULATION ───────────────────────────────────────────────────────

def simulate(config_path: str = None) -> pd.DataFrame:
    """
    Generate a synthetic event log with ~500K rows.
    Returns a DataFrame ready to be passed into data_pipeline.run_pipeline().
    """
    cfg = load_sim_config(config_path)
    rng = np.random.default_rng(cfg["random_seed"])

    start = pd.Timestamp(cfg["experiment"]["start_date"], tz="UTC")
    end   = pd.Timestamp(cfg["experiment"]["end_date"],   tz="UTC")
    sd    = cfg["session_duration"]

    # 1. User population
    users = generate_users(cfg, rng)

    # 2. Session counts
    session_counts = assign_session_counts(users, cfg, rng)

    # 3. Build event log
    all_events = []
    total_sessions = 0

    for _, user in users.iterrows():
        n_sess = session_counts[_]
        base_durations = sample_durations(n_sess, sd["base_scale"], sd["base_shape"], rng)
        durations = apply_duration_multipliers(base_durations, user, cfg, rng)
        timestamps = sample_timestamps(n_sess, start, end, rng)

        for sess_idx in range(n_sess):
            session_id = str(uuid.uuid4())
            events = generate_events_for_session(
                user_id       = user["user_id"],
                session_id    = session_id,
                variant       = user["variant"],
                session_start = pd.Timestamp(timestamps[sess_idx]),
                session_dur_sec = float(durations[sess_idx]),
                rng           = rng,
            )
            all_events.extend(events)
        total_sessions += n_sess

        if (_ + 1) % 10_000 == 0:
            logger.info(f"  {_ + 1:,}/{cfg['n_users']:,} users processed "
                        f"({len(all_events):,} events so far)")

    df = pd.DataFrame(all_events)
    logger.info(
        f"Simulation complete — {len(df):,} events, "
        f"{total_sessions:,} sessions, "
        f"{cfg['n_users']:,} users"
    )
    return df, users  # return users too for ground-truth checks


# ─── 8. GROUND-TRUTH VERIFICATION ────────────────────────────────────────────

def verify_simulation(df: pd.DataFrame, users: pd.DataFrame, cfg: dict) -> None:
    """
    Quick sanity checks — confirm the DGP injected the intended effect.
    Run this after simulate() to verify before handing off to the pipeline.
    """
    from scipy.stats import ttest_ind

    sessions = (
        df.groupby(["user_id", "session_id", "variant"])
        .agg(
            duration=("timestamp", lambda x: (x.max() - x.min()).total_seconds())
        )
        .reset_index()
    )

    ctrl = sessions[sessions["variant"] == "control"]["duration"]
    trt  = sessions[sessions["variant"] == "treatment"]["duration"]
    raw_lift = (trt.mean() - ctrl.mean()) / ctrl.mean()
    t, p = ttest_ind(ctrl, trt)

    logger.info("─── Ground-truth verification ───────────────────")
    logger.info(f"  Control   mean duration : {ctrl.mean():.1f}s")
    logger.info(f"  Treatment mean duration : {trt.mean():.1f}s")
    logger.info(f"  Raw lift                : {raw_lift:+.1%}  (injected: "
                f"{cfg['treatment_effect']['session_duration_lift']:+.1%})")
    logger.info(f"  t-stat={t:.3f}, p={p:.4e}")
    logger.info(f"  Power users in control  : "
                f"{users[users['variant']=='control']['is_power_user'].mean():.1%}")
    logger.info(f"  Power users in treatment: "
                f"{users[users['variant']=='treatment']['is_power_user'].mean():.1%}")
    logger.info("─────────────────────────────────────────────────")


# ─── 9. ENTRYPOINT ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    cfg = load_sim_config(config_path)

    df, users = simulate(config_path)

    # Verify the DGP before saving
    verify_simulation(df, users, cfg)

    # Save
    out_events = Path("data/raw/events.parquet")
    out_users  = Path("data/raw/users_ground_truth.parquet")
    out_events.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(out_events, index=False)
    users.to_parquet(out_users, index=False)
    logger.info(f"Saved events → {out_events}")
    logger.info(f"Saved ground truth → {out_users}")
