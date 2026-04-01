from src.data_pipeline      import run_pipeline, check_pre_experiment_balance
from src.causal_models      import psm_pipeline, ancova, summarise_model
from src.stats_tests        import run_significance_battery

# 1. Load the pipeline output
user_df = run_pipeline("data/raw/events.parquet")

covariates = ["tenure_days", "avg_pages_per_sess", "avg_events_per_sess"]

# 2. Balance BEFORE matching
print("=== Balance before PSM ===")
check_pre_experiment_balance(user_df, covariates)

# 3. PSM
matched_df = psm_pipeline(user_df, covariates, caliper=0.05)

# 4. Balance AFTER matching — SMDs should now be < 0.1
print("=== Balance after PSM ===")
check_pre_experiment_balance(matched_df, covariates)

# 5. ANCOVA on matched sample (covariate-adjusted, HC3 robust SEs)
model = ancova(matched_df, outcome="avg_session_dur", covariates=covariates)
print(summarise_model(model))

# 6. Full significance battery
battery = run_significance_battery(matched_df, outcome_col="avg_session_dur")
print(battery)
