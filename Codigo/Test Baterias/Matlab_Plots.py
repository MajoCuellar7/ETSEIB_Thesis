"""
plot_EKF_results.py
Plots EKF results from MATLAB in the thesis color scheme.
Run after runEKF_dual.m has exported EKF_dual_results.csv.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import AutoMinorLocator
import os

# ── PATHS ─────────────────────────────────────────────────────────────
# results_file  = "/Users/majocuellar/Documents/GitHub/Master_Thesis/Codigo/Test Baterias/Additional CSV/Charge_EKF_dual_results.csv"
# results_file  = "/Users/majocuellar/Documents/GitHub/Master_Thesis/Codigo/Test Baterias/Additional CSV/Pulses_EKF_dual_results.csv"
# results_file  = "/Users/majocuellar/Documents/GitHub/Master_Thesis/Codigo/Test Baterias/Additional CSV/Steps_EKF_dual_results.csv"
# graphs_folder = "/Users/majocuellar/Documents/GitHub/Master_Thesis/Codigo/Test Baterias/Battery_Graphs_Tests/EKF/dual"
results_file  = "/Users/majocuellar/Documents/GitHub/Master_Thesis/Codigo/Test Baterias/Additional CSV/Charge_EKF_soc_results.csv"
# results_file  = "/Users/majocuellar/Documents/GitHub/Master_Thesis/Codigo/Test Baterias/Additional CSV/Pulses_EKF_soc_results.csv"
# results_file  = "/Users/majocuellar/Documents/GitHub/Master_Thesis/Codigo/Test Baterias/Additional CSV/Steps_EKF_soc_results.csv"
graphs_folder = "/Users/majocuellar/Documents/GitHub/Master_Thesis/Codigo/Test Baterias/Battery_Graphs_Tests/EKF/soc"
tag           = "Charge_Test"   # ← change per test; used in filename
Q_nom_Ah      = 27.6160           # nominal Q [Ah]
R0_nom_mOhm   = 2.41            # median R0 from characterization [mΩ]

os.makedirs(graphs_folder, exist_ok=True)

# ── THESIS COLOR PALETTE ──────────────────────────────────────────────
color_v     = "#82D982"   # green  — primary / SOC EKF
color_c     = "#8282FF"   # blue   — secondary / CC reference
color_extra = "#FF82FF"   # pink   — bounds / third curve
color_v_l   = "#061406"   # dark label
color_c_l   = "#00001A"   # dark label

# ── LOAD ──────────────────────────────────────────────────────────────
df = pd.read_csv(results_file)
t_min    = df["time_s"] / 60
sochat   = df["sochat"] * 100
socbound = df["socbound"] * 100
soc_ref  = df["soc_ref"] * 100

# Detect whether this is a dual-EKF results file or SOC-only
is_dual = "R0hat_Ohm" in df.columns

if is_dual:
    R0hat    = df["R0hat_Ohm"] * 1000
    R0bound  = df["R0bound_Ohm"] * 1000
    R0_table = df["R0_table_Ohm"] * 1000
    Qhat     = df["Qhat_As"] / 3600
    Qbound   = df["Qbound_As"] / 3600
    print("Dual EKF results detected — plotting all 4 figures.")
else:
    print("SOC-only results detected — plotting figures 1 and 2 only.")
# ── HELPER: shared grid style ─────────────────────────────────────────
def style_ax(ax):
    ax.grid(True, which='major', linestyle="-",  alpha=0.6, color='gray')
    ax.grid(True, which='minor', linestyle="--", alpha=0.3, color='gray')
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))

# ═════════════════════════════════════════════════════════════════════
# FIGURE 1 — SOC estimation
# ═════════════════════════════════════════════════════════════════════
fig1, ax = plt.subplots(figsize=(12, 7))

ax.plot(t_min, sochat + socbound, color=color_extra, linewidth=1.0, linestyle='--', label="3σ bounds")
ax.plot(t_min, sochat - socbound, color=color_extra, linewidth=1.0, linestyle='--')
ax.plot(t_min, soc_ref,  color=color_c, linewidth=1.5, label="Coulomb counting")
ax.plot(t_min, sochat,   color=color_v, linewidth=2.0, label="EKF estimate")

ax.set_xlabel("Time (min)", fontweight='bold')
ax.set_ylabel("SOC [%]",    fontweight='bold', color=color_v_l)
ax.set_ylim(0, 110)
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
style_ax(ax)
ax.legend(frameon=True, shadow=True, loc='lower right')
fig1.tight_layout()

p1 = os.path.join(graphs_folder, f"EKF_SOC_{tag}.pdf")
fig1.savefig(p1, format='pdf', bbox_inches='tight')
print(f"Saved: {p1}")

# ═════════════════════════════════════════════════════════════════════
# FIGURE 2 — SOC error (EKF vs CC)
# ═════════════════════════════════════════════════════════════════════
fig2, ax = plt.subplots(figsize=(12, 5))

err = soc_ref - sochat
ax.plot(t_min,  socbound, color=color_extra, linewidth=1.5, linestyle='--', label="3σ bounds")
ax.plot(t_min, -socbound, color=color_extra, linewidth=1.5, linestyle='--')
ax.plot(t_min, err, color=color_c, linewidth=1.5, label="CC − EKF difference")

ax.set_xlabel("Time (min)",  fontweight='bold')
ax.set_ylabel("SOC error [%]", fontweight='bold', color=color_v_l)
ax.set_ylim(-10, 10)
ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
style_ax(ax)
ax.legend(frameon=True, shadow=True, loc='upper left')
fig2.tight_layout()

p2 = os.path.join(graphs_folder, f"EKF_SOC_error_{tag}.pdf")
fig2.savefig(p2, format='pdf', bbox_inches='tight')
print(f"Saved: {p2}")

# ═════════════════════════════════════════════════════════════════════
# FIGURE 3 — R0 estimate vs characterization table
# ═════════════════════════════════════════════════════════════════════
if is_dual:
    fig3, ax = plt.subplots(figsize=(12, 5))

    ax.plot(t_min, R0hat + R0bound, color=color_extra, linewidth=1.5, linestyle='--', label="3σ bounds")
    ax.plot(t_min, R0hat - R0bound, color=color_extra, linewidth=1.5, linestyle='--')
    ax.plot(t_min, R0_table, color=color_c,   linewidth=1.5, label="Characterization table (SOC-matched)")
    ax.plot(t_min, R0hat,    color=color_v,   linewidth=2.0,
        label="R₀ estimate (SOH filter)")

    ax.set_xlabel("Time (min)", fontweight='bold')
    ax.set_ylabel("R₀ [mΩ]",    fontweight='bold', color=color_v_l)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    style_ax(ax)
    ax.legend(frameon=True, shadow=True)
    fig3.tight_layout()

    p3 = os.path.join(graphs_folder, f"EKF_R0_{tag}.pdf")
    fig3.savefig(p3, format='pdf', bbox_inches='tight')
    print(f"Saved: {p3}")

# ═════════════════════════════════════════════════════════════════════
# FIGURE 4 — Q estimation
# ═════════════════════════════════════════════════════════════════════
if is_dual:
    fig4, ax = plt.subplots(figsize=(12, 5))

    ax.plot(t_min, Qhat + Qbound, color=color_extra, linewidth=1.5, linestyle='--', label="3σ bounds")
    ax.plot(t_min, Qhat - Qbound, color=color_extra, linewidth=1.5, linestyle='--')
    ax.axhline(Q_nom_Ah, color=color_c,   linewidth=1.5, label=f"Nominal Q = {Q_nom_Ah:.3f} Ah")
    ax.plot(t_min, Qhat, color=color_v, linewidth=2.0, label="Q estimate (SOH filter)")

    ax.set_xlabel("Time (min)", fontweight='bold')
    ax.set_ylabel("Capacity Q [Ah]", fontweight='bold', color=color_v_l)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    style_ax(ax)
    ax.legend(frameon=True, shadow=True)
    fig4.tight_layout()

    p4 = os.path.join(graphs_folder, f"EKF_Q_{tag}.pdf")
    fig4.savefig(p4, format='pdf', bbox_inches='tight')
    print(f"Saved: {p4}")

plt.close('all')
print("\nAll EKF plots saved.")