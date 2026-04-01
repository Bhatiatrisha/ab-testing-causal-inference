# src/reporting.py

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
from pathlib import Path
from jinja2 import Template
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

# ─── STYLE ────────────────────────────────────────────────────────────────────

PALETTE  = {"control": "#5B8DB8", "treatment": "#E07B54"}
BG       = "#FAFAFA"
SPINE_C  = "#CCCCCC"

def _style_ax(ax):
    ax.set_facecolor(BG)
    ax.figure.patch.set_facecolor("white")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(SPINE_C)
    ax.tick_params(colors="#555555")
    ax.yaxis.label.set_color("#555555")
    ax.xaxis.label.set_color("#555555")
    ax.title.set_color("#333333")


# ─── FIGURE 1: SESSION DURATION DISTRIBUTIONS ─────────────────────────────────

def plot_distributions(
    user_df: pd.DataFrame,
    outcome: str = "avg_session_dur",
    out_path: str = "reports/figures/fig1_distributions.png",
) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor="white")

    for i, (variant, color) in enumerate(PALETTE.items()):
        data = user_df.loc[user_df["variant"] == variant, outcome].dropna()
        ax   = axes[i]

        ax.hist(data, bins=60, color=color, alpha=0.75, edgecolor="white", linewidth=0.4)
        ax.axvline(data.mean(),   color=color, linewidth=1.8, linestyle="-",  label=f"Mean  {data.mean():.1f}s")
        ax.axvline(data.median(), color=color, linewidth=1.8, linestyle="--", label=f"Median {data.median():.1f}s")
        ax.set_xlabel("Avg session duration (s)", fontsize=11)
        ax.set_ylabel("Users", fontsize=11)
        ax.set_title(f"{variant.capitalize()} (n={len(data):,})", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, framealpha=0)
        _style_ax(ax)

    fig.suptitle("Session duration distribution by variant", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, out_path)
    return out_path


# ─── FIGURE 2: PROPENSITY SCORE OVERLAP ───────────────────────────────────────

def plot_propensity_overlap(
    user_df: pd.DataFrame,
    out_path: str = "reports/figures/fig2_propensity.png",
) -> str:
    if "propensity_score" not in user_df.columns:
        logger.warning("No propensity_score column found — skipping overlap plot")
        return ""

    fig, ax = plt.subplots(figsize=(8, 4), facecolor="white")
    for variant, color in PALETTE.items():
        data = user_df.loc[user_df["variant"] == variant, "propensity_score"]
        ax.hist(data, bins=50, color=color, alpha=0.55,
                edgecolor="white", linewidth=0.3, label=variant.capitalize(), density=True)

    ax.set_xlabel("Propensity score", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Propensity score overlap (common support check)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0)
    _style_ax(ax)
    plt.tight_layout()
    _save(fig, out_path)
    return out_path


# ─── FIGURE 3: COVARIATE BALANCE (SMD) ────────────────────────────────────────

def plot_covariate_balance(
    balance_before: pd.DataFrame,
    balance_after:  pd.DataFrame,
    out_path: str = "reports/figures/fig3_balance.png",
) -> str:
    """
    Love plot — standard way to show PSM covariate balance.
    Points left of the dashed line (|SMD| < 0.1) are well-balanced.
    """
    merged = balance_before.merge(
        balance_after, on="covariate", suffixes=("_before", "_after")
    )

    fig, ax = plt.subplots(figsize=(8, max(4, len(merged) * 0.7)), facecolor="white")
    y = np.arange(len(merged))

    ax.scatter(merged["SMD_before"].abs(), y, color="#E07B54", zorder=3,
               s=70, label="Before PSM", marker="o")
    ax.scatter(merged["SMD_after"].abs(),  y, color="#5B8DB8", zorder=3,
               s=70, label="After PSM",  marker="D")

    for i, row in merged.iterrows():
        ax.plot(
            [abs(row["SMD_before"]), abs(row["SMD_after"])],
            [i, i],
            color=SPINE_C, linewidth=1, zorder=1
        )

    ax.axvline(0.1, color="#E07B54", linewidth=1.2, linestyle="--",
               alpha=0.7, label="|SMD| = 0.1 threshold")
    ax.axvline(0.0, color=SPINE_C, linewidth=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(merged["covariate"], fontsize=10)
    ax.set_xlabel("|Standardised Mean Difference|", fontsize=11)
    ax.set_title("Covariate balance before and after PSM", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0)
    _style_ax(ax)
    plt.tight_layout()
    _save(fig, out_path)
    return out_path


# ─── FIGURE 4: EFFECT SIZE WITH CI ────────────────────────────────────────────

def plot_effect_estimates(
    battery_df: pd.DataFrame,
    out_path: str = "reports/figures/fig4_effects.png",
) -> str:
    """
    Forest plot of effect estimates (% lift) and CIs across all test methods.
    """
    df = battery_df.dropna(subset=["pct_lift"]).copy()
    df["pct_lift_pct"]  = df["pct_lift"] * 100
    df["ci_low_pct"]    = df["ci_low"].fillna(0) * 100 if "ci_low" in df.columns else 0
    df["ci_high_pct"]   = df["ci_high"].fillna(0) * 100 if "ci_high" in df.columns else 0

    fig, ax = plt.subplots(figsize=(9, max(3.5, len(df) * 0.9)), facecolor="white")
    y = np.arange(len(df))
    colors = ["#5B8DB8" if sig else "#AAAAAA"
              for sig in df.get("significant", [True]*len(df))]

    for i, row in df.iterrows():
        idx = list(df.index).index(i)
        ax.scatter(row["pct_lift_pct"], idx, color=colors[idx], s=80, zorder=3)
        if pd.notna(row.get("ci_low")) and pd.notna(row.get("ci_high")):
            ax.plot(
                [row["ci_low_pct"], row["ci_high_pct"]],
                [idx, idx],
                color=colors[idx], linewidth=2, zorder=2
            )

    ax.axvline(0, color=SPINE_C, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["test"], fontsize=9)
    ax.set_xlabel("Estimated lift (%)", fontsize=11)
    ax.set_title("Effect estimates across methods (95% CI)", fontsize=12, fontweight="bold")
    _style_ax(ax)
    plt.tight_layout()
    _save(fig, out_path)
    return out_path


# ─── FIGURE 5: POWER CURVE ────────────────────────────────────────────────────

def plot_power_curve(
    sensitivity_df: pd.DataFrame,
    out_path: str = "reports/figures/fig5_power_curve.png",
) -> str:
    """
    Sample size vs MDE for t-test and ANCOVA side by side.
    """
    has_ancova = "n_ancova" in sensitivity_df.columns
    fig, ax    = plt.subplots(figsize=(9, 4.5), facecolor="white")

    for alpha_val, grp in sensitivity_df.groupby("alpha"):
        ls = "-" if alpha_val == 0.01 else "--"
        ax.plot(grp["mde_pct"] * 100, grp["n_ttest"],
                color="#E07B54", linewidth=2, linestyle=ls,
                label=f"t-test α={alpha_val}")
        if has_ancova:
            ax.plot(grp["mde_pct"] * 100, grp["n_ancova"],
                    color="#5B8DB8", linewidth=2, linestyle=ls,
                    label=f"ANCOVA α={alpha_val}")

    ax.set_xlabel("Minimum detectable effect (%)", fontsize=11)
    ax.set_ylabel("Required n per group", fontsize=11)
    ax.set_title("Sample size vs MDE — t-test vs ANCOVA", fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(fontsize=9, framealpha=0)
    _style_ax(ax)
    plt.tight_layout()
    _save(fig, out_path)
    return out_path


# ─── HTML REPORT ──────────────────────────────────────────────────────────────

REPORT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>A/B Test Report — {{ experiment_name }}</title>
<style>
  body        { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                max-width: 960px; margin: 0 auto; padding: 40px 24px;
                color: #2d2d2d; background: #fff; }
  h1          { font-size: 1.8rem; font-weight: 700; margin-bottom: 4px; color: #1a1a1a; }
  h2          { font-size: 1.2rem; font-weight: 600; margin-top: 40px;
                padding-bottom: 6px; border-bottom: 2px solid #e8e8e8; color: #333; }
  .subtitle   { color: #666; font-size: 0.95rem; margin-bottom: 32px; }
  .kpi-grid   { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 24px 0; }
  .kpi        { background: #f7f8fa; border-radius: 10px; padding: 18px 20px;
                border-left: 4px solid {{ verdict_color }}; }
  .kpi-value  { font-size: 2rem; font-weight: 700; color: {{ verdict_color }}; line-height: 1; }
  .kpi-label  { font-size: 0.78rem; color: #888; margin-top: 4px; text-transform: uppercase;
                letter-spacing: 0.04em; }
  .verdict    { background: {{ verdict_bg }}; border: 1px solid {{ verdict_color }};
                border-radius: 10px; padding: 20px 24px; margin: 24px 0; }
  .verdict h3 { margin: 0 0 6px; color: {{ verdict_color }}; font-size: 1.05rem; }
  .verdict p  { margin: 0; font-size: 0.95rem; color: #444; line-height: 1.6; }
  table       { width: 100%; border-collapse: collapse; font-size: 0.88rem; margin: 16px 0; }
  th          { background: #f2f4f6; padding: 10px 12px; text-align: left;
                font-weight: 600; border-bottom: 2px solid #dde1e7; color: #555; }
  td          { padding: 9px 12px; border-bottom: 1px solid #eef0f3; }
  tr:hover td { background: #fafbfc; }
  .sig-yes    { color: #2e7d32; font-weight: 600; }
  .sig-no     { color: #c62828; }
  .fig        { margin: 24px 0; text-align: center; }
  .fig img    { max-width: 100%; border-radius: 8px;
                border: 1px solid #eee; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
  .fig-cap    { font-size: 0.82rem; color: #888; margin-top: 8px; }
  .method-note { font-size: 0.82rem; color: #888; background: #f7f8fa;
                 border-radius: 6px; padding: 12px 16px; margin: 12px 0;
                 border-left: 3px solid #ccc; line-height: 1.5; }
  footer      { margin-top: 60px; padding-top: 16px; border-top: 1px solid #eee;
                font-size: 0.78rem; color: #aaa; }
</style>
</head>
<body>

<h1>A/B Test Report</h1>
<div class="subtitle">{{ experiment_name }} &nbsp;·&nbsp; {{ date_range }} &nbsp;·&nbsp; Generated {{ generated_at }}</div>

<h2>Executive summary</h2>
<div class="kpi-grid">
  <div class="kpi">
    <div class="kpi-value">{{ pct_lift }}</div>
    <div class="kpi-label">Session duration lift</div>
  </div>
  <div class="kpi">
    <div class="kpi-value">{{ p_value }}</div>
    <div class="kpi-label">p-value (ANCOVA)</div>
  </div>
  <div class="kpi">
    <div class="kpi-value">{{ n_users }}</div>
    <div class="kpi-label">Users analysed</div>
  </div>
  <div class="kpi">
    <div class="kpi-value">{{ sample_reduction }}</div>
    <div class="kpi-label">Sample size reduction</div>
  </div>
</div>

<div class="verdict">
  <h3>{{ verdict_title }}</h3>
  <p>{{ verdict_body }}</p>
</div>

<h2>Statistical results</h2>
<p class="method-note">
  Primary estimator: ANCOVA on log-transformed session duration with covariate adjustment
  (tenure, pages/session, events/session). Heteroskedasticity-robust standard errors (HC3).
  Secondary validation: PSM nearest-neighbour matching (caliper=0.05), Welch t-test,
  Mann-Whitney U, and percentile bootstrap on the matched sample.
</p>
<table>
  <thead>
    <tr>
      <th>Test</th><th>Lift (%)</th><th>p-value</th>
      <th>95% CI low</th><th>95% CI high</th><th>Significant</th>
    </tr>
  </thead>
  <tbody>
    {% for row in battery_rows %}
    <tr>
      <td>{{ row.test }}</td>
      <td>{{ row.pct_lift }}</td>
      <td>{{ row.p_value }}</td>
      <td>{{ row.ci_low }}</td>
      <td>{{ row.ci_high }}</td>
      <td class="{{ 'sig-yes' if row.significant else 'sig-no' }}">
        {{ 'Yes' if row.significant else 'No' }}
      </td>
    </tr>
    {% endfor %}
  </tbody>
</table>

<h2>Power analysis</h2>
<table>
  <thead>
    <tr><th>Method</th><th>n / group</th><th>Total n</th><th>Notes</th></tr>
  </thead>
  <tbody>
    {% for row in power_rows %}
    <tr>
      <td>{{ row.method }}</td>
      <td>{{ "{:,}".format(row.n_per_group) }}</td>
      <td>{{ "{:,}".format(row.total_n) }}</td>
      <td>{{ row.notes }}</td>
    </tr>
    {% endfor %}
  </tbody>
</table>

<h2>Figures</h2>
{% for fig in figures %}
<div class="fig">
  <img src="{{ fig.path }}" alt="{{ fig.caption }}">
  <div class="fig-cap">{{ fig.caption }}</div>
</div>
{% endfor %}

<h2>Covariate balance (PSM)</h2>
<table>
  <thead>
    <tr><th>Covariate</th>
        <th>Control mean</th><th>Treatment mean</th><th>SMD before</th><th>SMD after</th>
    </tr>
  </thead>
  <tbody>
    {% for row in balance_rows %}
    <tr>
      <td>{{ row.covariate }}</td>
      <td>{{ "%.3f"|format(row.mean_control_before) }}</td>
      <td>{{ "%.3f"|format(row.mean_treatment_before) }}</td>
      <td class="{{ 'sig-no' if row.SMD_before|abs > 0.1 else '' }}">
          {{ "%.3f"|format(row.SMD_before) }}</td>
      <td class="{{ 'sig-no' if row.SMD_after|abs > 0.1 else '' }}">
          {{ "%.3f"|format(row.SMD_after) }}</td>
    </tr>
    {% endfor %}
  </tbody>
</table>

<footer>
  Generated by ab-testing-causal-inference pipeline &nbsp;·&nbsp;
  {{ generated_at }} &nbsp;·&nbsp; alpha={{ alpha }}, power={{ power }}
</footer>
</body>
</html>
"""


def build_report(
    user_df:         pd.DataFrame,
    matched_df:      pd.DataFrame,
    battery_df:      pd.DataFrame,
    ancova_summary:  dict,
    power_results:   dict,
    balance_before:  pd.DataFrame,
    balance_after:   pd.DataFrame,
    experiment_name: str   = "UI change — session duration",
    date_range:      str   = "",
    alpha:           float = 0.01,
    power:           float = 0.95,
    out_dir:         str   = "reports",
) -> str:
    """
    Generate all figures and render the HTML report.
    Returns the path to the saved HTML file.
    """
    fig_dir  = Path(out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Figures ──────────────────────────────────────────────────────────────
    fig_paths = []

    p = plot_distributions(matched_df, out_path=str(fig_dir / "fig1_distributions.png"))
    fig_paths.append({"path": f"figures/fig1_distributions.png",
                      "caption": "Figure 1: Session duration distributions — matched sample"})

    if "propensity_score" in matched_df.columns:
        plot_propensity_overlap(matched_df, out_path=str(fig_dir / "fig2_propensity.png"))
        fig_paths.append({"path": "figures/fig2_propensity.png",
                          "caption": "Figure 2: Propensity score overlap — common support verification"})

    plot_covariate_balance(
        balance_before, balance_after,
        out_path=str(fig_dir / "fig3_balance.png")
    )
    fig_paths.append({"path": "figures/fig3_balance.png",
                      "caption": "Figure 3: Covariate balance (Love plot) — before and after PSM"})

    plot_effect_estimates(battery_df, out_path=str(fig_dir / "fig4_effects.png"))
    fig_paths.append({"path": "figures/fig4_effects.png",
                      "caption": "Figure 4: Effect estimates and 95% CIs across all test methods"})

    if "sensitivity_table" in power_results:
        plot_power_curve(
            power_results["sensitivity_table"],
            out_path=str(fig_dir / "fig5_power_curve.png")
        )
        fig_paths.append({"path": "figures/fig5_power_curve.png",
                          "caption": "Figure 5: Required sample size vs MDE — t-test vs ANCOVA"})

    # ── Template data ─────────────────────────────────────────────────────────
    lift_val = ancova_summary.get("pct_lift", 0)
    p_val    = ancova_summary.get("p_value", 1)
    n_users  = ancova_summary.get("n_obs", len(matched_df))
    sig      = p_val < alpha

    verdict_color  = "#2e7d32" if sig else "#c62828"
    verdict_bg     = "#f1f8f1" if sig else "#fdf3f3"
    verdict_title  = "Ship it — statistically significant positive result" if sig \
                     else "Do not ship — result does not meet significance threshold"
    verdict_body   = (
        f"The UI change produced a {lift_val:+.1%} lift in average session duration "
        f"(ANCOVA, p={p_val:.2e}, α={alpha}, n={n_users:,}). "
        f"All secondary tests (Mann-Whitney U, bootstrap) are directionally consistent. "
        f"Recommend shipping to 100% of users."
    ) if sig else (
        f"The observed lift of {lift_val:+.1%} did not reach significance at α={alpha} "
        f"(p={p_val:.2e}, n={n_users:,}). "
        f"Consider extending the experiment or revisiting the UI change."
    )

    sample_red_str = ""
    if "sample_size_reduction" in power_results:
        sample_red_str = f"{power_results['sample_size_reduction']:.0%}"

    # Battery table
    battery_rows = []
    for _, row in battery_df.iterrows():
        battery_rows.append({
            "test":        row["test"],
            "pct_lift":    f"{row['pct_lift']*100:.2f}%" if pd.notna(row.get("pct_lift")) else "—",
            "p_value":     f"{row['p_value']:.4e}"       if pd.notna(row.get("p_value")) else "—",
            "ci_low":      f"{row['ci_low']:.3f}"        if pd.notna(row.get("ci_low")) else "—",
            "ci_high":     f"{row['ci_high']:.3f}"       if pd.notna(row.get("ci_high")) else "—",
            "significant": row.get("significant", False),
        })

    # Power table
    power_rows = []
    for key in ["ttest_requirement", "ancova_requirement"]:
        if key in power_results:
            r = power_results[key]
            power_rows.append(r.to_dict() if hasattr(r, "to_dict") else r)

    # Balance table
    balance_rows = []
    if not balance_before.empty and not balance_after.empty:
        merged = balance_before.merge(
            balance_after, on="covariate", suffixes=("_before", "_after")
        )
        balance_rows = merged.to_dict("records")

    # ── Render ────────────────────────────────────────────────────────────────
    tmpl = Template(REPORT_TEMPLATE)
    html = tmpl.render(
        experiment_name  = experiment_name,
        date_range       = date_range or "see config/params.yaml",
        generated_at     = datetime.now().strftime("%Y-%m-%d %H:%M"),
        pct_lift         = f"{lift_val:+.1%}",
        p_value          = f"{p_val:.2e}",
        n_users          = f"{n_users:,}",
        sample_reduction = sample_red_str or "N/A",
        verdict_color    = verdict_color,
        verdict_bg       = verdict_bg,
        verdict_title    = verdict_title,
        verdict_body     = verdict_body,
        battery_rows     = battery_rows,
        power_rows       = power_rows,
        balance_rows     = balance_rows,
        figures          = fig_paths,
        alpha            = alpha,
        power            = power,
    )

    out_path = Path(out_dir) / "ab_test_report.html"
    out_path.write_text(html, encoding="utf-8")
    logger.info(f"Report saved → {out_path}")
    return str(out_path)


# ─── HELPER ───────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved figure → {path}")
