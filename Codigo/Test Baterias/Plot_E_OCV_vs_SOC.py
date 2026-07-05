"""
Generates:
  Figure A - E (extracted OCV) vs SOC, oriented 0% -> 100% (same style as OCV-SOC curve)
  Figure B - Combined plot: OCV-SOC (green) and E-SOC (lilac) sharing the SOC axis,
             Y axis labeled "Voltage [V]"

Data sources:
  - battery_param_vs_SOC_Param_0A_10A_30A_Jun30.xlsx  -> columns SOC_pct, E_V
  - OCV_SOC_table_Pulsos_10A_3A_1A_Jun22.xlsx          -> columns
        "SOC % (Q = 27.616 Ah — Q_ref / discharge test)", "OCV [V]"
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import AutoMinorLocator

# ─────────────────────────── COLOR PALETTE ───────────────────────────────────
# Light theme — matches the rest of the thesis graphs
color_v     = "#82D982"   # green   — voltage line / Q_ref OCV curve
color_c     = "#8282FF"   # blue    — current line
color_extra = "#FF82FF"   # magenta — Q_test OCV curve (second line)
color_v_l   = "#061406"   # dark label for voltage y-axis
color_c_l   = "#00001A"   # dark label for current y-axis
color_E     = "#C39BD3"   # lilac   — extracted E (OCV) vs SOC line

# ─────────────────────────── FILE PATHS ───────────────────────────────────
param_path = "/Users/majocuellar/Documents/GitHub/Master_Thesis/Codigo/Test Baterias/Additional CSV/battery_param_vs_SOC_Param_0A_10A_30A_Jun30.xlsx"
ocv_path   = "/Users/majocuellar/Documents/GitHub/Master_Thesis/Codigo/Test Baterias/Additional CSV/OCV_SOC_table_Pulsos_10A_3A_1A_Jun22.xlsx"

out_dir = "/Users/majocuellar/Documents/GitHub/Master_Thesis/Codigo/Test Baterias/Battery_Graphs_Tests"
os.makedirs(out_dir, exist_ok=True)

E_vs_SOC_plot_path        = os.path.join(out_dir, "E_vs_SOC_curve_Jun30.pdf")
OCV_E_combined_plot_path  = os.path.join(out_dir, "OCV_and_E_vs_SOC_combined.pdf")

Q_ref = 27.616

# ─────────────────────────── STYLE HELPER ─────────────────────────────────
def style_ax(ax):
    ax.set_facecolor("white")
    ax.grid(which='major', linestyle='--', linewidth=0.7, color='0.75', zorder=0)
    ax.grid(which='minor', linestyle=':', linewidth=0.5, color='0.88', zorder=0)
    for spine in ax.spines.values():
        spine.set_color('0.3')
        spine.set_linewidth(0.8)
    ax.tick_params(which='both', colors='0.2')


# ─────────────────────────── LOAD DATA ─────────────────────────────────────
param_df = pd.read_excel(param_path)
param_df = param_df[['SOC_pct', 'E_V']].sort_values('SOC_pct').reset_index(drop=True)

ocv_df = pd.read_excel(ocv_path)
soc_ref_col = "SOC % (Q = 27.616 Ah — Q_ref / discharge test)"
ocv_df = ocv_df[[soc_ref_col, 'OCV [V]']].sort_values(soc_ref_col).reset_index(drop=True)


# ─────────────────────────── FIGURE A: E vs SOC (0% -> 100%) ───────────────
print("Generating Figure A – Extracted E (OCV) vs SOC, Q_ref orientation …")
figA, axA = plt.subplots(figsize=(9, 6))

axA.plot(param_df['SOC_pct'], param_df['E_V'],
         color=color_E, linewidth=2.0, marker='o', markersize=6, zorder=4,
         label=f'E, Q = {Q_ref} Ah')

axA.set_xlabel('SOC [%]', fontweight='bold')
axA.set_ylabel('E [V]', fontweight='bold', color=color_v_l)

axA.xaxis.set_major_locator(ticker.MultipleLocator(10))
axA.xaxis.set_minor_locator(ticker.MultipleLocator(5))
axA.set_xlim(-5, 105)

axA.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
axA.yaxis.set_minor_locator(AutoMinorLocator(2))
axA.set_ylim(2.9, 4.25)

style_ax(axA)
axA.legend(frameon=True, shadow=True, loc='lower right')
figA.tight_layout()

figA.savefig(E_vs_SOC_plot_path, format='pdf', bbox_inches='tight')


# ─────────────────────────── FIGURE B: Combined OCV + E vs SOC ─────────────
print("Generating Figure B – Combined OCV-SOC (green) and E-SOC (lilac) …")
figB, axB = plt.subplots(figsize=(9, 6))

axB.plot(ocv_df[soc_ref_col], ocv_df['OCV [V]'],
         color=color_v, linewidth=2.0, marker='o', markersize=6, zorder=4,
         label=f'OCV vs SOC (Q = {Q_ref} Ah)')

axB.plot(param_df['SOC_pct'], param_df['E_V'],
         color=color_E, linewidth=2.0, marker='o', markersize=6, zorder=4,
         label=f'E vs SOC')

axB.set_xlabel('SOC [%]', fontweight='bold')
axB.set_ylabel('Voltage [V]', fontweight='bold', color=color_v_l)

axB.xaxis.set_major_locator(ticker.MultipleLocator(10))
axB.xaxis.set_minor_locator(ticker.MultipleLocator(5))
axB.set_xlim(-5, 105)

axB.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
axB.yaxis.set_minor_locator(AutoMinorLocator(2))
axB.set_ylim(2.9, 4.25)

style_ax(axB)
axB.legend(frameon=True, shadow=True, loc='lower right')
figB.tight_layout()

figB.savefig(OCV_E_combined_plot_path, format='pdf', bbox_inches='tight')

print("Done.")
print("Saved:", E_vs_SOC_plot_path)
print("Saved:", OCV_E_combined_plot_path)