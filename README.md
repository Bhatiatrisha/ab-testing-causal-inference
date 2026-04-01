# A/B Testing & Causal Inference Framework

A production-grade pipeline for causal effect estimation on large-scale
user interaction data. Implements difference-in-differences, propensity
score matching, and automated power analysis on 500K+ event logs.

**Key results (simulated dataset)**
- Quantified a UI change's causal effect on session duration: **+12% lift, p < 0.01**
- Reduced required sample size by **22%** at 95% power via ANCOVA covariate adjustment
- Validated results across four estimators: DiD, PSM, Welch t-test, bootstrap CI

---

## Repository structure
```
ab-testing-causal-inference/
├── src/
│   ├── simulation.py       # Synthetic event log generator (500K rows, configurable confounders)
│   ├── data_pipeline.py    # Ingestion, cleaning, session aggregation, SRM check
│   ├── causal_models.py    # PSM, difference-in-differences, ANCOVA
│   ├── stats_tests.py      # Welch t-test, Mann-Whitney, bootstrap CI, CUPED
│   ├── power_analysis.py   # Sample size calculator, MDE sweep, runtime estimator
│   └── reporting.py        # Matplotlib figures + Jinja2 HTML executive report
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   ├── 02_causal_analysis.ipynb  # Full pipeline walkthrough with narrative
│   └── 03_power_analysis.ipynb   # Sample size trade-off analysis
├── data/
│   ├── raw/                # Simulated or real event logs (gitignored)
│   └── processed/          # Pipeline outputs (gitignored)
├── reports/
│   ├── ab_test_report.html # Auto-generated executive report
│   └── figures/            # PNG plots produced by reporting.py
├── tests/
│   ├── test_data_pipeline.py
│   ├── test_causal_models.py
│   ├── test_stats_tests.py
│   └── test_power_analysis.py
├── config/
│   └── params.yaml         # Experiment dates, alpha, power, covariate list
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions: pytest + flake8 on every push
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quickstart

**1. Clone and install**
```bash
git clone https://github.com/your-username/ab-testing-causal-inference.git
cd ab-testing-causal-inference
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Generate synthetic data**
```bash
python src/simulation.py config/params.yaml
```

This writes `data/raw/events.parquet` (≈500K events) and
`data/raw/users_ground_truth.parquet`. The ground-truth file lets you verify
the pipeline recovers the injected treatment effect.

**3. Run the pipeline**
```bash
python src/data_pipeline.py
```

**4. Run the full analysis and generate the report**

Open `notebooks/02_causal_analysis.ipynb` and run all cells.
The HTML report is saved to `reports/ab_test_report.html`.

**5. Run tests**
```bash
pytest tests/ -v
```

---

## Methods

### Propensity score matching (PSM)
Logistic regression estimates P(treatment | covariates). Nearest-neighbour
matching with caliper = 0.05 creates a balanced matched sample. Balance is
verified using standardised mean differences (SMD < 0.1 threshold).

### Difference-in-differences (DiD)
Panel regression with a post × treated interaction term. Standard errors
are clustered by user. Requires pre-experiment outcome data.

### ANCOVA
OLS on log-transformed session duration with covariate adjustment and HC3
heteroskedasticity-robust standard errors. The primary estimator — the
log transformation handles right skew and makes the coefficient directly
interpretable as % lift.

### Power analysis
Sample size is computed via the two-sample t-test formula and the
ANCOVA-adjusted formula `n_ancova = n_ttest × (1 − R²)`, where R² comes
from the fitted ANCOVA model. This is the source of the 22% sample size
reduction — covariates explaining ~25% of outcome variance tighten the
residual enough to reduce the required n at constant power.

---

## Configuration

All tunable parameters live in `config/params.yaml`:
```yaml
experiment:
  start_date: "2024-01-15"
  end_date:   "2024-02-15"
  expected_split: 0.5

cleaning:
  winsor_lower: 0.01
  winsor_upper: 0.99

stats:
  alpha: 0.01
  power: 0.95

covariates:
  - tenure_days
  - avg_pages_per_sess
  - avg_events_per_sess
```

---

## Requirements
```
pandas>=2.0
numpy>=1.26
scipy>=1.11
scikit-learn>=1.3
statsmodels>=0.14
matplotlib>=3.7
seaborn>=0.12
jinja2>=3.1
pyyaml>=6.0
pyarrow>=14.0
pytest>=7.4
```

---

## .gitignore

Add this so raw data and generated outputs are never committed:
```
data/raw/
data/processed/
reports/figures/
reports/*.html
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
*.egg-info/
.DS_Store
```

---

## Roadmap

- [ ] Bayesian A/B testing module (PyMC)
- [ ] Sequential testing / always-valid p-values
- [ ] Segment-level heterogeneous treatment effects (HTE)
- [ ] Streamlit dashboard for interactive report

---

## License

MIT
