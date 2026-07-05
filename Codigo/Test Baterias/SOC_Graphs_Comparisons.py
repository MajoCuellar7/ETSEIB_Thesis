"""
plot_EKF_comparison.py
Compares Coulomb counting, SOC-only EKF, and dual EKF SOC estimates
on the same axes, for a single test profile.

Run AFTER both runEKF.m and runEKF_dual.m have exported their results
for the same test (i.e. both {tag}_EKF_soc_results.csv and
{tag}_EKF_dual_results.csv must exist).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import AutoMinorLocator
import os

# ── PATHS ─────────────────────────────────────────────────────────────
base_folder   = "/Users/majocuellar/Documents/GitHub/Master_Thesis/Codigo/Test Baterias/Additional CSV"
graphs_folder = "/Users/majocuellar/Documents/GitHub/Master_Thesis/Codigo/Test Baterias/Battery_Graphs_Tests/EKF/comparison"

tag = "Charge_Test"   # ← change per test: "Charge_Test", "Pulses_Test", "Steps_Test"

soc_only_file = os.path.join(base_folder, f"{tag.replace('_Test','')}_EKF_soc_results.csv")
dual_file     = os.path.join(base_folder, f"{tag.replace('_Test','')}_EKF_dual_results.csv")

os.makedirs(graphs_folder, exist_ok=True)

# ── THESIS COLOR PALETTE ──────────────────────────────────────────────
color_cc    = "#8282FF"   # blue   — Coulomb counting reference
color_soc   = "#FF82FF"   # pink   — SOC-only EKF
color_dual  = "#82D982"   # green  — dual EKF
color_v_l   = "#061406"   # dark label

def style_ax(ax):
    ax.grid(True, which='major', linestyle="-",  alpha=0.6, color='gray')
    ax.grid(True, which='minor', linestyle="--", alpha=0.3, color='gray')
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))

# ── LOAD BOTH FILES ───────────────────────────────────────────────────
df_soc  = pd.read_csv(soc_only_file)
df_dual = pd.read_csv(dual_file)

t_min_soc   = df_soc["time_s"] / 60
sochat_soc  = df_soc["sochat"] * 100
soc_ref_soc = df_soc["soc_ref"] * 100

t_min_dual   = df_dual["time_s"] / 60
sochat_dual  = df_dual["sochat"] * 100
soc_ref_dual = df_dual["soc_ref"] * 100

# Sanity check: both runs should share the same underlying test data,
# so the CC reference should match. Warn if it doesn't.
if len(t_min_soc) != len(t_min_dual):
    print(f"⚠️  Warning: SOC-only ({len(t_min_soc)} samples) and dual "
          f"({len(t_min_dual)} samples) results have different lengths. "
          f"Check that both were run on the same test file.")

# ═════════════════════════════════════════════════════════════════════
# FIGURE 1 — SOC comparison: CC vs SOC-only EKF vs Dual EKF
# ═════════════════════════════════════════════════════════════════════
fig1, ax = plt.subplots(figsize=(12, 7))

ax.plot(t_min_dual, soc_ref_dual, color=color_cc,   linewidth=1.8,
        linestyle='-',  label="Coulomb counting")
ax.plot(t_min_soc,  sochat_soc,   color=color_soc,  linewidth=1.5,
        linestyle='-', label="EKF estimate (SOC Filter)")
ax.plot(t_min_dual, sochat_dual,  color=color_dual, linewidth=1.8,
        linestyle='-',  label="Dual EKF estimate (SOC&SOH Filter)")

ax.set_xlabel("Time (min)", fontweight='bold')
ax.set_ylabel("SOC [%]",    fontweight='bold', color=color_v_l)
ax.set_ylim(0, 110)
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
style_ax(ax)
ax.legend(frameon=True, shadow=True, loc='lower right')
fig1.tight_layout()

p1 = os.path.join(graphs_folder, f"EKF_SOC_comparison_{tag}.pdf")
fig1.savefig(p1, format='pdf', bbox_inches='tight')
print(f"Saved: {p1}")

# ═════════════════════════════════════════════════════════════════════
# FIGURE 2 — SOC error comparison: (CC - SOC-only) vs (CC - dual)
# ═════════════════════════════════════════════════════════════════════
fig2, ax = plt.subplots(figsize=(12, 5))

err_soc  = soc_ref_soc  - sochat_soc
err_dual = soc_ref_dual - sochat_dual

ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
ax.plot(t_min_soc,  err_soc,  color=color_soc,  linewidth=1.8,
        linestyle='-', label="EKF estimate (SOC Filter)")
ax.plot(t_min_dual, err_dual, color=color_dual, linewidth=1.8,
        linestyle='-', label="Dual EKF estimate (SOC&SOH Filter)")

ax.set_xlabel("Time (min)",     fontweight='bold')
ax.set_ylabel("SOC error [%]",  fontweight='bold', color=color_v_l)
ax.set_ylim(-10, 10)
ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
style_ax(ax)
ax.legend(frameon=True, shadow=True, loc='upper left')
fig2.tight_layout()

p2 = os.path.join(graphs_folder, f"EKF_SOC_error_comparison_{tag}.pdf")
fig2.savefig(p2, format='pdf', bbox_inches='tight')
print(f"Saved: {p2}")

# ── PRINT RMSE FOR QUICK REFERENCE (matches your existing MATLAB output) ──
rmse_soc  = np.sqrt(np.mean(err_soc**2))
rmse_dual = np.sqrt(np.mean(err_dual**2))
print(f"\nRMSE (SOC-only):  {rmse_soc:.4f} %")
print(f"RMSE (dual EKF):   {rmse_dual:.4f} %")
print(f"Difference:        {rmse_dual - rmse_soc:+.4f} percentage points")

plt.close('all')
print("\nComparison plots saved.")