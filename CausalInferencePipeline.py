from src.data_pipeline  import run_pipeline, check_pre_experiment_balance
from src.causal_models  import psm_pipeline, compute_propensity_scores, ancova, summarise_model
from src.stats_tests    import run_significance_battery
from src.power_analysis import auto_power_analysis
from src.reporting      import build_report

# ── 1. Pipeline ───────────────────────────────────────────────────────────────
user_df = run_pipeline("data/raw/events.parquet")
covariates = ["tenure_days", "avg_pages_per_sess", "avg_events_per_sess"]

# ── 2. Balance before ─────────────────────────────────────────────────────────
balance_before = check_pre_experiment_balance(user_df, covariates)

# ── 3. PSM ────────────────────────────────────────────────────────────────────
matched_df = psm_pipeline(user_df, covariates, caliper=0.05)

# ── 4. Balance after ──────────────────────────────────────────────────────────
balance_after = check_pre_experiment_balance(matched_df, covariates)

# ── 5. ANCOVA on matched sample ───────────────────────────────────────────────
model   = ancova(matched_df, outcome="avg_session_dur", covariates=covariates)
summary = summarise_model(model)

# ── 6. Significance battery ───────────────────────────────────────────────────
battery = run_significance_battery(matched_df, outcome_col="avg_session_dur")

# ── 7. Power analysis (pass R² from ANCOVA to get sample size reduction) ──────
power_results = auto_power_analysis(
    user_df,
    outcome    = "avg_session_dur",
    mde_pct    = 0.10,
    alpha      = 0.01,
    power      = 0.95,
    r_squared  = model.rsquared,   # <-- this is where the 22% reduction comes from
)

# ── 8. Report ─────────────────────────────────────────────────────────────────
report_path = build_report(
    user_df         = user_df,
    matched_df      = matched_df,
    battery_df      = battery,
    ancova_summary  = summary,
    power_results   = power_results,
    balance_before  = balance_before,
    balance_after   = balance_after,
    experiment_name = "UI change — session duration",
    date_range      = "2024-01-15 to 2024-02-15",
    alpha           = 0.01,
    power           = 0.95,
)
print(f"Report: {report_path}")
