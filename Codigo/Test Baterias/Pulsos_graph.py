"""
Pulse Test Analysis — Welion SHP270-30 NMC+ Cell
Generates:
  1. Voltage & Current vs Time  (anomalous spike removed)
  2. OCV–SOC curve  (two lines: Q_ref and Q_test)
  3. OCV–SOC curve  (single line: Q_ref only)
  4. OCV–SOC lookup table (.xlsx)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import AutoMinorLocator

# ─────────────────────────── USER PARAMETERS ─────────────────────────────────
file_path     = "/Users/majocuellar/Documents/GitHub/Master_Thesis/Codigo/Test Baterias/Raw/Pulsos_10A_3A_1A_Jun22.CSV"
excel_folder  = "/Users/majocuellar/Documents/GitHub/Master_Thesis/Codigo/Test Baterias/Additional CSV"
graphs_folder = "/Users/majocuellar/Documents/GitHub/Master_Thesis/Codigo/Test Baterias/Battery_Graphs_Tests"

os.makedirs(excel_folder,  exist_ok=True)
os.makedirs(graphs_folder, exist_ok=True)

excel_base_name   = os.path.splitext(os.path.basename(file_path))[0]
excel_output_path  = os.path.join(excel_folder,  f"OCV_SOC_table_{excel_base_name}.xlsx")
pulse_plot_path    = os.path.join(graphs_folder, f"PulseCurve_{excel_base_name}.pdf")
ocv_plot_path      = os.path.join(graphs_folder, f"OCV_SOC_curve_{excel_base_name}.pdf")
ocv_ref_plot_path  = os.path.join(graphs_folder, f"OCV_SOC_curve_Qref_{excel_base_name}.pdf")

# ── SOC anchor voltages  (adjust freely) ─────────────────────────────────────
V_SOC_100 = 4.135    # [V] voltage defined as SOC = 100%
V_SOC_0   = 3.021    # [V] voltage defined as SOC = 0%

# ── Capacity reference (adjust freely) ───────────────────────────────────────
Q_ref = 27.616       # [Ah]  from separate discharge test; used for SOC_ref column

# ── Signal-processing knobs ──────────────────────────────────────────────────
BASELINE_I = 0.230   # [A]  microcontroller quiescent current (added during "rest")
PULSE_THR  = 1.0     # [A]  threshold separating rest from real pulse
LOW_THR    = 0.5     # [A]  lower bound to detect the 0.8 A phase

# ── Anomalous-spike window to remove ────────────────────────────────────────
ANOM_S = 17897       # first sample of spurious spike (inclusive)
ANOM_E = 17912       # last  sample of spurious spike (inclusive)

# ─────────────────────────── COLOR PALETTE ───────────────────────────────────
# Light theme — matches the rest of the thesis graphs
color_v     = "#82D982"   # green  — voltage line / Q_ref OCV curve
color_c     = "#8282FF"   # blue   — current line
color_extra = "#FF82FF"   # magenta — Q_test OCV curve (second line)
color_v_l   = "#061406"   # dark label for voltage y-axis
color_c_l   = "#00001A"   # dark label for current y-axis

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading data …")

CSV_LOAD_PATH = file_path
_SANDBOX = "/mnt/user-data/uploads/Pulsos_10A_3A_1A_Jun22.CSV"
if not os.path.isfile(CSV_LOAD_PATH) and os.path.isfile(_SANDBOX):
    print(f"[info] Local path not found; using sandbox copy: {_SANDBOX}")
    CSV_LOAD_PATH = _SANDBOX

cols = ['TIME[s]', 'UAVG[V]', 'P[W]', 'IAVG[A]',
        'WH[Wh]', 'AH[Ah]', 'WHP[Wh]', 'AHP[Ah]', 'WHM[Wh]', 'AHM[Ah]', 'Timestamp']
df    = pd.read_csv(CSV_LOAD_PATH, sep=';', skiprows=18, names=cols)
df['t_s'] = np.arange(len(df))
I_raw = df['IAVG[A]'].values.copy()
V_raw = df['UAVG[V]'].values.copy()
t     = df['t_s'].values

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  REMOVE ANOMALOUS SPIKE
# ═══════════════════════════════════════════════════════════════════════════════
I_clean = I_raw.copy()
I_clean[ANOM_S:ANOM_E + 1] = BASELINE_I
print(f"Spike removed (t = {ANOM_S}–{ANOM_E} s) → replaced with {BASELINE_I} A baseline.")

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  COULOMB COUNTING
# ═══════════════════════════════════════════════════════════════════════════════
dt_s         = 1.0
Ah_cumul     = np.cumsum(I_clean) * dt_s / 3600

Q_test_total = float(Ah_cumul[-1])
print(f"\nQ discharged during pulse test  (incl. µC baseline): {Q_test_total:.4f} Ah")
print(f"Q_ref  (from discharge test):                         {Q_ref:.3f} Ah")

SOC_test = (1 - Ah_cumul / Q_test_total) * 100
SOC_ref  = (1 - Ah_cumul / Q_ref)        * 100

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  OCV EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════
is_pulse = (I_clean > PULSE_THR).astype(int)
rise_idx = np.where(np.diff(is_pulse) > 0)[0]

ocv_records = []

pulse_labels = (
    [f'Before 10A pulse #{n + 1}' for n in range(18)] +
    ['Before 1st 3A pulse', 'Before 2nd 3A pulse']
)

valid_rise = [r for r in rise_idx if t[r] != ANOM_S]

for n, idx in enumerate(valid_rise):
    label = pulse_labels[n] if n < len(pulse_labels) else f'Before pulse #{n + 1}'
    ocv_records.append({
        'Label'       : label,
        't_s'         : int(t[idx]),
        'OCV_V'       : float(V_raw[idx]),
        'Ah_consumed' : float(Ah_cumul[idx]),
        'SOC_test_%'  : float(SOC_test[idx]),
        'SOC_ref_%'   : float(SOC_ref[idx]),
    })

# ── Before 0.8 A phase ───────────────────────────────────────────────────────
is_08         = (I_clean > LOW_THR) & (I_clean < 2.0)
indices_after = np.where(t > 14670)[0]
first_08_rel  = np.where(is_08[indices_after])[0]
t_08_start    = int(t[indices_after[first_08_rel[0]]])
idx_pre08     = t_08_start - 1
ocv_records.append({
    'Label'       : 'Before 0.8A phase',
    't_s'         : int(t[idx_pre08]),
    'OCV_V'       : float(V_raw[idx_pre08]),
    'Ah_consumed' : float(Ah_cumul[idx_pre08]),
    'SOC_test_%'  : float(SOC_test[idx_pre08]),
    'SOC_ref_%'   : float(SOC_ref[idx_pre08]),
})

# ── Minimum voltage ──────────────────────────────────────────────────────────
last_clean = ANOM_S - 1
ocv_records.append({
    'Label'       : 'Minimum voltage',
    't_s'         : last_clean,
    'OCV_V'       : float(V_raw[last_clean]),
    'Ah_consumed' : float(Ah_cumul[last_clean]),
    'SOC_test_%'  : float(SOC_test[last_clean]),
    'SOC_ref_%'   : float(SOC_ref[last_clean]),
})

ocv_df = (pd.DataFrame(ocv_records)
          .sort_values('SOC_ref_%', ascending=True)
          .reset_index(drop=True))

# ── Anchor 100 % SOC to V_SOC_100 ────────────────────────────────────────────
# The coulomb-counting SOC already places the first measured rest at 100 %,
# but the measured voltage there (~4.074 V) is lower than V_SOC_100 because
# some charge was lost before the test started. We override the OCV and SOC
# of that first point with the user-defined anchor values so the curve reaches
# V_SOC_100 at exactly 100 %.
first_idx = ocv_df['SOC_ref_%'].idxmax()
ocv_df.loc[first_idx, 'OCV_V']      = V_SOC_100
ocv_df.loc[first_idx, 'SOC_ref_%']  = 100.0
ocv_df.loc[first_idx, 'SOC_test_%'] = 100.0
ocv_df.loc[first_idx, 'Ah_consumed'] = 0.0

# ── Print table ───────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"{'OCV – SOC Table':^80}")
print(f"{'='*80}")
print(f"{'Label':<38} {'OCV [V]':>7} {'SOC_test%':>10} {'SOC_ref%':>10}")
print(f"{'-'*80}")
for _, row in ocv_df.iterrows():
    print(f"{row['Label']:<38} {row['OCV_V']:>7.4f} "
          f"{row['SOC_test_%']:>10.2f} {row['SOC_ref_%']:>10.2f}")
print(f"{'='*80}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  SAVE EXCEL
# ═══════════════════════════════════════════════════════════════════════════════
excel_cols = ['Label', 't_s', 'OCV_V', 'Ah_consumed', 'SOC_test_%', 'SOC_ref_%']
excel_out  = ocv_df[excel_cols].copy()
excel_out.columns = [
    'Label', 'Time [s]', 'OCV [V]', 'Ah consumed',
    f'SOC % (Q = {Q_test_total:.3f} Ah — pulse test)',
    f'SOC % (Q = {Q_ref} Ah — Q_ref / discharge test)',
]

_EXCEL_OUT = excel_output_path
if not os.path.isdir(os.path.dirname(excel_output_path)):
    _EXCEL_OUT = "/mnt/user-data/outputs/OCV_SOC_table.xlsx"
    os.makedirs("/mnt/user-data/outputs", exist_ok=True)

excel_out.to_excel(_EXCEL_OUT, index=False, sheet_name='OCV_SOC')
print(f"\nOCV–SOC table saved → {_EXCEL_OUT}")


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: light-theme grid
# ═══════════════════════════════════════════════════════════════════════════════
def style_ax(ax):
    ax.grid(True, which='major', linestyle="-",  alpha=0.6, color='gray')
    ax.grid(True, which='minor', linestyle="--", alpha=0.3, color='gray')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Voltage & Current vs Time
# ═══════════════════════════════════════════════════════════════════════════════
print("\nGenerating Figure 1 – V & I vs Time …")
fig1, ax1 = plt.subplots(figsize=(12, 7))

ax1.set_xlabel('Continuous Time (seconds)', fontweight='bold')
ax1.set_ylabel('Voltage (UAVG [V])', color=color_v_l, fontweight='bold')
ax1.plot(t, V_raw, color=color_v, linewidth=2.0, label='Voltage (V)')
ax1.tick_params(axis='y', labelcolor=color_v_l)
ax1.set_ylim(2.9, 4.3)
ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
style_ax(ax1)

ax2 = ax1.twinx()
ax2.set_ylabel('Current (IAVG [A])', color=color_c_l, fontweight='bold')
ax2.plot(t, I_clean, color=color_c, linewidth=2.0, alpha=0.85, label='Current (A) Absolute')
ax2.tick_params(axis='y', labelcolor=color_c_l)
ax2.set_ylim(0, max(I_clean) * 1.2)

l1, lb1 = ax1.get_legend_handles_labels()
l2, lb2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, lb1 + lb2, loc='upper right', frameon=True, shadow=True)

ax1.set_xlim(t[0], t[-1])
fig1.tight_layout()

_P1 = pulse_plot_path
if not os.path.isdir(os.path.dirname(pulse_plot_path)):
    _P1 = "/mnt/user-data/outputs/PulseCurve_Pulsos_10A_3A_1A_Jun22.pdf"
    os.makedirs("/mnt/user-data/outputs", exist_ok=True)

fig1.savefig(_P1, format='pdf', bbox_inches='tight')
print(f"Figure 1 saved → {_P1}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — OCV–SOC Curve  (two lines: Q_ref and Q_test)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 2 – OCV–SOC curve (two lines) …")
fig2, ax3 = plt.subplots(figsize=(9, 6))

curve = ocv_df.sort_values('SOC_ref_%')

ax3.plot(curve['SOC_ref_%'], curve['OCV_V'],
         color=color_v, linewidth=2.0, marker='o', markersize=6, zorder=4,
         label=f'Q = {Q_ref} Ah')

ax3.plot(curve['SOC_test_%'], curve['OCV_V'],
         color=color_extra, linewidth=2.0, marker='s', markersize=5,
         linestyle='--', zorder=3,
         label=f'Q = {Q_test_total:.3f} Ah')

ax3.set_xlabel('SOC [%]', fontweight='bold')
ax3.set_ylabel('OCV [V]', fontweight='bold', color=color_v_l)
# ax3.set_title('OCV vs SOC — Welion SHP270-30 (pulse test)')

ax3.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax3.xaxis.set_minor_locator(ticker.MultipleLocator(5))
ax3.set_xlim(-5, 105)

ax3.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax3.yaxis.set_minor_locator(AutoMinorLocator(2))
ax3.set_ylim(2.9, 4.25)

style_ax(ax3)
ax3.legend(frameon=True, shadow=True, loc='lower right')
fig2.tight_layout()

_P2 = ocv_plot_path
if not os.path.isdir(os.path.dirname(ocv_plot_path)):
    _P2 = "/mnt/user-data/outputs/OCV_SOC_curve_Pulsos_10A_3A_1A_Jun22.pdf"

fig2.savefig(_P2, format='pdf', bbox_inches='tight')
print(f"Figure 2 saved → {_P2}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — OCV–SOC Curve  (single line: Q_ref only)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 3 – OCV–SOC curve (Q_ref only) …")
fig3, ax4 = plt.subplots(figsize=(9, 6))

ax4.plot(curve['SOC_ref_%'], curve['OCV_V'],
         color=color_v, linewidth=2.0, marker='o', markersize=6, zorder=4,
         label=f'Q = {Q_ref} Ah')

ax4.set_xlabel('SOC [%]', fontweight='bold')
ax4.set_ylabel('OCV [V]', fontweight='bold', color=color_v_l)
# ax4.set_title('OCV vs SOC — Welion SHP270-30 (pulse test)')

ax4.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax4.xaxis.set_minor_locator(ticker.MultipleLocator(5))
ax4.set_xlim(-5, 105)

ax4.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax4.yaxis.set_minor_locator(AutoMinorLocator(2))
ax4.set_ylim(2.9, 4.25)

style_ax(ax4)
ax4.legend(frameon=True, shadow=True, loc='lower right')
fig3.tight_layout()

_P3 = ocv_ref_plot_path
if not os.path.isdir(os.path.dirname(ocv_ref_plot_path)):
    _P3 = "/mnt/user-data/outputs/OCV_SOC_curve_Qref_Pulsos_10A_3A_1A_Jun22.pdf"

fig3.savefig(_P3, format='pdf', bbox_inches='tight')
print(f"Figure 3 saved → {_P3}")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  V_SOC_100                           : {V_SOC_100} V")
print(f"  V_SOC_0                             : {V_SOC_0} V")
print(f"  Q_ref  (discharge test)             : {Q_ref:.3f} Ah")
print(f"  Q_test (this pulse test)            : {Q_test_total:.4f} Ah")
print(f"  OCV curve points                    : {len(ocv_df)}")
print(f"  SOC range (Q_ref)  : "
      f"{curve['SOC_ref_%'].min():.1f} % – {curve['SOC_ref_%'].max():.1f} %")
print(f"  OCV range          : "
      f"{curve['OCV_V'].min():.4f} V – {curve['OCV_V'].max():.4f} V")
print("=" * 60)

plt.close('all')
print("\nDone ✓")

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from matplotlib.ticker import MultipleLocator, AutoMinorLocator

# # =============================================================================
# # 0. SETTINGS
# # =============================================================================
# file_path     = "/Users/majocuellar/Documents/GitHub/Master_Thesis/Codigo/Test Baterias/Pulsos_10A_3A_1A_Jun22.CSV"
# excel_folder  = "/Users/majocuellar/Documents/GitHub/Master_Thesis/Codigo/Test Baterias/Additional CSV"
# graphs_folder = "/Users/majocuellar/Documents/GitHub/Master_Thesis/Codigo/Test Baterias/Battery_Graphs_Tests"

# os.makedirs(excel_folder, exist_ok=True)
# os.makedirs(graphs_folder, exist_ok=True)

# excel_base_name    = os.path.splitext(os.path.basename(file_path))[0]
# excel_output_path  = os.path.join(excel_folder, f"OCV_SOC_table_{excel_base_name}.xlsx")
# pulse_plot_path    = os.path.join(graphs_folder, f"PulseCurve_{excel_base_name}.pdf")
# ocv_plot_path      = os.path.join(graphs_folder, f"OCV_SOC_curve_{excel_base_name}.pdf")

# V_SOC_100 = 4.141     # [V] OCV defined as SOC = 100% (e.g. fully-relaxed post-charge voltage)
# V_SOC_0   = 3.021     # [V] OCV defined as SOC = 0%   (e.g. your discharge cutoff voltage)
# Q_ref     = 27.596    # [Ah] capacity used to convert Ah discharged -> SOC. Change this freely
#                        # (e.g. to your separate discharge-test Q, or to this test's own Ah-at-cutoff)

# MIN_BLOCK_LEN = 5       # samples (~s) -> filters single-sample noise spikes
# REST_I_THRESH = 0.5     # [A] below this = "rest"
# LOW_I_THRESH  = 1.5     # [A] boundary between "low" (~0.8A) pulses and "3A" pulses
# MED_I_THRESH  = 6.0     # [A] boundary between "3A" pulses and "10A" pulses

# # Color palette (shared across both plots)
# color_v     = "#82D982"   # voltage / "low" pulses
# color_c     = "#8282FF"   # current / "10A" pulses
# color_v_l   = "#061406"   # dark voltage label
# color_c_l   = "#00001A"   # dark current label
# color_extra = "#FF82FF"   # "3A" pulses / accent

# # =============================================================================
# # 1. LOAD
# # =============================================================================
# df = pd.read_csv(file_path, sep=';', comment='#', skip_blank_lines=True, encoding='utf-8')
# df.columns = df.columns.str.strip()

# for c in ["UAVG[V]", "IAVG[A]", "AH[Ah]", "P[W]"]:
#     df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '.'), errors="coerce")

# df = df.drop_duplicates(subset=["TIME[s]"], keep="first").reset_index(drop=True)
# df["idx"] = df.index                      # 1 sample ~= 1 second (Logging Interval = 1.000 s)
# df["AH[Ah]"] = df["AH[Ah]"].fillna(0.0)    # first few rows before the logger settles

# # =============================================================================
# # 2. CLASSIFY EACH SAMPLE BY CURRENT LEVEL AND GROUP INTO BLOCKS
# # =============================================================================
# def classify(i):
#     if i < REST_I_THRESH:
#         return "rest"
#     elif i < LOW_I_THRESH:
#         return "0.8A"
#     elif i < MED_I_THRESH:
#         return "3A"
#     else:
#         return "10A"

# df["level"] = df["IAVG[A]"].apply(classify)
# df["block"] = (df["level"] != df["level"].shift()).cumsum()

# blocks = (df.groupby("block")
#             .agg(level=("level", "first"),
#                  start=("idx", "first"),
#                  end=("idx", "last"),
#                  n=("idx", "count"),
#                  V_first=("UAVG[V]", "first"),
#                  V_last=("UAVG[V]", "last"),
#                  AH_first=("AH[Ah]", "first"),
#                  AH_last=("AH[Ah]", "last"))
#             .reset_index(drop=True))

# blocks_real = blocks[blocks["n"] >= MIN_BLOCK_LEN].reset_index(drop=True)

# def soc_from_ah(ah):
#     return 100 * (1 - ah / Q_ref)

# # Ah accumulated at the moment voltage first reaches V_SOC_0 (informational,
# # used only to explain/compare against your chosen Q_ref - NOT used in the SOC formula)
# below_cutoff = df[df["UAVG[V]"] <= V_SOC_0]
# Q_at_cutoff_this_test = below_cutoff["AH[Ah]"].iloc[0] if not below_cutoff.empty else np.nan
# print(f"Ah discharged when V first reached {V_SOC_0} V in THIS test: {Q_at_cutoff_this_test:.3f} Ah")
# print(f"Q_ref used for SOC conversion: {Q_ref:.3f} Ah")
# print(f"-> Difference: {Q_at_cutoff_this_test - Q_ref:+.3f} Ah "
#       f"({100*(Q_at_cutoff_this_test - Q_ref)/Q_ref:+.2f} % of Q_ref)\n")

# # =============================================================================
# # 3. FOR EACH REAL PULSE BLOCK, GRAB THE OCV AT THE END OF THE PRECEDING REST
# # =============================================================================
# records = []
# pulse_counters = {"10A": 0, "3A": 0, "0.8A": 0}

# for i in range(1, len(blocks_real)):
#     cur = blocks_real.loc[i]
#     prev = blocks_real.loc[i - 1]
#     if cur["level"] != "rest" and prev["level"] == "rest":
#         pulse_counters[cur["level"]] += 1
#         ah_at_ocv = prev["AH_last"]
#         records.append({
#             "pulse_type": cur["level"],
#             "pulse_number": pulse_counters[cur["level"]],
#             "sample_idx_OCV": int(prev["end"]),
#             "OCV_V": prev["V_last"],
#             "AH_discharged": ah_at_ocv,
#             "SOC_%": soc_from_ah(ah_at_ocv),
#         })

# ocv_table = pd.DataFrame(records).sort_values("sample_idx_OCV").reset_index(drop=True)

# # --- Override the very first point with your chosen 100% SOC anchor ---
# # (the measured first-rest OCV is kept in a separate column for transparency)
# ocv_table.loc[0, "measured_OCV_V"] = ocv_table.loc[0, "OCV_V"]
# ocv_table.loc[0, "OCV_V"] = V_SOC_100
# ocv_table.loc[0, "AH_discharged"] = 0.0
# ocv_table.loc[0, "SOC_%"] = 100.0

# # --- Add the 0% SOC anchor explicitly, using your chosen V_SOC_0 ---
# ocv_table.loc[len(ocv_table)] = {
#     "pulse_type": "Minimum Voltage",
#     "pulse_number": 1,
#     "sample_idx_OCV": int(below_cutoff["idx"].iloc[0]) if not below_cutoff.empty else np.nan,
#     "OCV_V": V_SOC_0,
#     "AH_discharged": Q_at_cutoff_this_test,
#     "SOC_%": soc_from_ah(Q_at_cutoff_this_test),
#     "measured_OCV_V": np.nan,
# }

# ocv_table = ocv_table.sort_values("sample_idx_OCV").reset_index(drop=True)
# print(ocv_table.to_string(index=False))

# # =============================================================================
# # 4. SAVE TABLE AS EXCEL
# # =============================================================================
# ocv_table.to_excel(excel_output_path, index=False, sheet_name="OCV_SOC")
# print(f"\nSaved OCV-SOC table to: {excel_output_path}")

# # =============================================================================
# # 5. PLOT 1 - OCV vs SOC (same palette as the pulse plot)
# # =============================================================================
# fig1, ax = plt.subplots(figsize=(9, 6))
# plot_colors = {
#     "10A": color_c,         # blue
#     "3A": color_extra,      # magenta/pink accent
#     "0.8A": color_v,         # green
#     "Minimum Voltage": color_c_l  # near-black
# }
# for ptype, sub in ocv_table.groupby("pulse_type"):
#     ax.scatter(sub["SOC_%"], sub["OCV_V"], label=ptype, color=plot_colors.get(ptype, "gray"),
#                s=60, zorder=3, edgecolor=color_v_l, linewidth=0.4)
# ax.plot(ocv_table["SOC_%"], ocv_table["OCV_V"], color="gray", alpha=0.4, zorder=1)
# ax.set_xlabel("SOC [%]", fontweight="bold")
# ax.set_ylabel("OCV [V]", fontweight="bold", color=color_v_l)
# ax.set_title("OCV vs SOC — Welion SHP270-30 (pulse test)")
# ax.invert_xaxis()
# ax.grid(True, which='major', linestyle="-", alpha=0.6, color='gray')
# ax.grid(True, which='minor', linestyle="--", alpha=0.3, color='gray')
# ax.legend(frameon=True, shadow=True)
# fig1.tight_layout()
# fig1.savefig(ocv_plot_path, format="pdf", bbox_inches="tight")
# print(f"Saved OCV-SOC plot to: {ocv_plot_path}")

# # =============================================================================
# # 6. PLOT 2 - VOLTAGE & CURRENT vs CONTINUOUS TIME (pulse overview)
# # =============================================================================
# valid_data_mask = (df["UAVG[V]"] > 0) & (df["IAVG[A]"].abs() > 0.1)
# df_clean = df[valid_data_mask].copy().reset_index(drop=True)
# df_clean["Continuous_Time[s]"] = df_clean.index

# fig2, ax1 = plt.subplots(figsize=(12, 7))

# ax1.set_xlabel("Continuous Time (seconds)", fontweight='bold')
# ax1.set_ylabel("Voltage (UAVG [V])", color=color_v_l, fontweight='bold')
# ax1.plot(
#     df_clean["Continuous_Time[s]"],
#     df_clean["UAVG[V]"],
#     color=color_v,
#     label="Voltage (V)",
#     linewidth=2.0
# )
# ax1.tick_params(axis="y", labelcolor=color_v_l)

# ax1.set_ylim(2.9, 4.3)
# ax1.yaxis.set_major_locator(MultipleLocator(0.1))
# ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
# ax1.grid(True, which='major', linestyle="-", alpha=0.6, color='gray')
# ax1.grid(True, which='minor', linestyle="--", alpha=0.3, color='gray')

# ax2 = ax1.twinx()

# max_current = df_clean["IAVG[A]"].abs().max()
# if pd.isna(max_current) or max_current == 0:
#     max_current = 20

# ax2.set_ylim(0, max_current * 1.2)

# ax2.set_ylabel("Current (IAVG [A])", color=color_c_l, fontweight='bold')
# ax2.plot(
#     df_clean["Continuous_Time[s]"],
#     df_clean["IAVG[A]"].abs(),
#     color=color_c,
#     label="Current (A) Absolute",
#     alpha=0.85
# )
# ax2.tick_params(axis="y", labelcolor=color_c_l)

# lines_1, labels_1 = ax1.get_legend_handles_labels()
# lines_2, labels_2 = ax2.get_legend_handles_labels()

# ax1.legend(
#     lines_1 + lines_2,
#     labels_1 + labels_2,
#     loc="upper right",
#     frameon=True,
#     shadow=True
# )

# fig2.tight_layout()
# fig2.savefig(pulse_plot_path, format="pdf", bbox_inches="tight")
# print(f"Saved pulse overview plot to: {pulse_plot_path}")

# plt.show()