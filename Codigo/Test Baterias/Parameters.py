"""
Battery Parameter Identification - Algebraic Method, BATCH version
====================================================================
Reads ONE continuous recording containing many repeated
rest -> 7A(90s) -> 20A(90s) -> [optional stepping pulse] -> rest
cycles, auto-detects every cycle, computes the SOC at each cycle
(via Coulomb counting), extracts the 2nd-order Thevenin ECM
parameters for each cycle, and plots how they change with SOC.

Based on: Devarakonda & Hu, Journal of Power Sources 268 (2014) 928-940
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.interpolate import interp1d
import os

# ─────────────────────────────────────────────────
#  ASK WHICH Q TO USE AS THE 100% SOC REFERENCE
# ─────────────────────────────────────────────────
print("Which capacity (Q, in Ah) do you want to use as the 100% SOC reference?")
print("  e.g. 27.596  -> from your separate discharge characterization test")
print("  e.g. 28.287  -> Ah discharged to your 3.021V cutoff in the pulse test")
_q_input = input("Q_ref (Ah) [default 27.596]: ").strip()
Q_REF = float(_q_input) if _q_input else 27.596
print(f"Using Q_ref = {Q_REF} Ah\n")

# ─────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────
# FILE_PATH      = "/Users/majocuellar/Documents/GitHub/Master_Thesis/Codigo/Test Baterias/Raw/Param_0A_7A_20A_Jun29.CSV"
FILE_PATH      = "/Users/majocuellar/Documents/GitHub/Master_Thesis/Codigo/Test Baterias/Raw/Param_0A_10A_30A_Jun30.CSV"
GRAPHS_FOLDER  = "/Users/majocuellar/Documents/GitHub/Master_Thesis/Codigo/Test Baterias/Battery_Graphs_Tests"
RESULTS_FOLDER = "/Users/majocuellar/Documents/GitHub/Master_Thesis/Codigo/Test Baterias/Additional CSV"

JUMP_THRESH  = 1.5   # [A] a current change bigger than this marks a new block (rest/phase1/phase2/stepping)
REST_THRESH  = 1.0   # [A] below this = "rest"
MIN_BLOCK_LEN = 5    # samples - filters single-sample noise spikes
MIN_PHASE_LEN = 30   # samples - phase1/phase2 must be at least this long to count as a real plateau
MAX_COV       = 0.05 # max coefficient of variation (std/mean) allowed within a phase1/phase2 block
                      # -> automatically rejects cycles like the one with the bad current-clamp reading

T_MIN, T_MAX, T_STEPS = 0.5, 60.0, 120  # T sweep range; paper constraint T<=T0/4 is enforced per-cycle below

# Color palette (consistent with your other thesis plots)
color_v     = "#82D982" # Green
color_c     = "#8282FF" # Lilac
color_v_l   = "#061406" # Super Dark Green
color_c_l   = "#00001A" # Super Dark Lilac
color_extra = "#FF82FF" # Pink
# ─────────────────────────────────────────────────


def load_data(filepath):
    print(f"Loading: {filepath}")
    all_cols = ['TIME[s]', 'UAVG[V]', 'P[W]', 'IAVG[A]', 'WH[Wh]', 'AH[Ah]',
                'WHP[Wh]', 'AHP[Ah]', 'WHM[Wh]', 'AHM[Ah]', 'Timestamp']
    df = pd.read_csv(filepath, sep=';', skiprows=16, names=all_cols,
                     encoding='utf-8', on_bad_lines='skip')
    for col in ['UAVG[V]', 'IAVG[A]', 'AH[Ah]']:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
    df = df.dropna(subset=['UAVG[V]', 'IAVG[A]']).reset_index(drop=True)
    print(f"  Rows: {len(df)}")
    # 1 Hz logging (confirmed by '#Logging Interval[s];1.000' in the file header) -> sample index == seconds
    t = np.arange(len(df), dtype=float)
    v = df['UAVG[V]'].to_numpy(dtype=float)
    i = np.abs(df['IAVG[A]'].to_numpy(dtype=float))   # abs(): robust to either sign convention
    ah = df['AH[Ah]'].fillna(0.0).to_numpy(dtype=float)
    print(f"  Duration: {t[-1]/3600:.2f} h   Voltage: {v.min():.3f}-{v.max():.3f} V   Current: {i.min():.3f}-{i.max():.3f} A")
    return t, v, i, ah


def find_cycles(i, ah):
    """
    Segment the WHOLE current trace into blocks by detecting jumps > JUMP_THRESH,
    classify each block as 'rest' or 'loaded', then group consecutive loaded
    blocks (between two rests) into cycles. Each cycle's first loaded block is
    phase 1, second is phase 2; any further block (e.g. a SOC-stepping pulse)
    is recorded but excluded from the 2-level parameter extraction.

    A cycle is marked invalid (and skipped) if phase 1 or phase 2 is too short
    or too noisy (high coefficient of variation) -- this is what automatically
    rejects a cycle where, e.g., the current clamp glitched.
    """
    n = len(i)
    di = np.abs(np.diff(i))
    block_id = np.zeros(n, dtype=int)
    block_id[1:] = np.cumsum(di > JUMP_THRESH)
    starts = np.where(np.diff(block_id, prepend=block_id[0] - 1) != 0)[0]
    ends = np.append(starts[1:], n)

    blocks = []
    for s, e in zip(starts, ends):
        seg = i[s:e]
        blocks.append({'start': int(s), 'end': int(e), 'n': int(e - s),
                       'mean': float(seg.mean()), 'std': float(seg.std())})
    blocks = pd.DataFrame(blocks)
    blocks['cov'] = blocks['std'] / blocks['mean'].replace(0, np.nan)
    blocks = blocks[blocks['n'] >= MIN_BLOCK_LEN].reset_index(drop=True)
    blocks['is_rest'] = blocks['mean'] < REST_THRESH

    cycles = []
    rest_ends = []   # last index (exclusive) of the REST block immediately before each cycle
    k = 0
    last_rest_end = None
    while k < len(blocks):
        if blocks.loc[k, 'is_rest']:
            last_rest_end = blocks.loc[k, 'end']
            k += 1
            continue
        sub = []
        while k < len(blocks) and not blocks.loc[k, 'is_rest']:
            sub.append(blocks.loc[k])
            k += 1
        cycles.append(sub)
        rest_ends.append(last_rest_end)

    parsed = []
    for cyc_num, (sub, rest_end) in enumerate(zip(cycles, rest_ends), 1):
        idx_start = sub[0]['start']
        n_sub = len(sub)
        valid = (
            n_sub >= 2
            and sub[0]['n'] >= MIN_PHASE_LEN and sub[1]['n'] >= MIN_PHASE_LEN
            and sub[0]['cov'] < MAX_COV and sub[1]['cov'] < MAX_COV
            and rest_end is not None and rest_end > 0
        )
        idx_T0 = sub[1]['start'] if n_sub >= 2 else None
        idx_end = sub[1]['end'] if n_sub >= 2 else None
        # FIX: use the last sample of the TRUE preceding rest block, not idx_start-1.
        # If the hardware's 0->I current step takes 2+ samples to ramp up, idx_start-1
        # can land on a mid-ramp sample (already partially loaded) instead of the
        # genuine rest voltage -- this silently corrupted R0 and E for any cycle with
        # a multi-sample ramp.
        idx_rest_ref = (rest_end - 1) if rest_end is not None else None
        ah_pre = ah[idx_rest_ref] if idx_rest_ref is not None and idx_rest_ref >= 0 else np.nan
        parsed.append(dict(
            cycle=cyc_num, valid=valid, idx_start=idx_start, idx_T0=idx_T0, idx_end=idx_end,
            idx_rest_ref=idx_rest_ref, n_subblocks=n_sub, AH_pre=ah_pre,
            I_meas=sub[0]['mean'] if n_sub >= 1 else np.nan,
            I1_meas=sub[1]['mean'] if n_sub >= 2 else np.nan,
            stepping_I=sub[2]['mean'] if n_sub >= 3 else np.nan,
            phase1_n=sub[0]['n'], phase2_n=sub[1]['n'] if n_sub >= 2 else np.nan,
            phase1_cov=sub[0]['cov'], phase2_cov=sub[1]['cov'] if n_sub >= 2 else np.nan,
            reject_reason=None,
        ))

    for p in parsed:
        if not p['valid']:
            reasons = []
            if p['n_subblocks'] < 2:
                reasons.append("no second current level found")
            else:
                if p['phase1_n'] < MIN_PHASE_LEN:
                    reasons.append(f"phase1 too short ({p['phase1_n']} samples)")
                if p['phase2_n'] < MIN_PHASE_LEN:
                    reasons.append(f"phase2 too short ({p['phase2_n']} samples)")
                if p['phase1_cov'] >= MAX_COV:
                    reasons.append(f"phase1 too noisy (cov={p['phase1_cov']:.3f})")
                if p['phase2_cov'] >= MAX_COV:
                    reasons.append(f"phase2 too noisy (cov={p['phase2_cov']:.3f})")
            p['reject_reason'] = "; ".join(reasons)

    return parsed


def extract_R0(v, i, idx_start, idx_rest_ref):
    v_before, v_after, I_applied = v[idx_rest_ref], v[idx_start], i[idx_start]
    if abs(I_applied) < 1e-6:
        return np.nan
    return abs(v_before - v_after) / abs(I_applied)


def interp_v(t_data, v_data, t_query):
    f = interp1d(t_data, v_data, kind='linear', fill_value='extrapolate')
    return float(f(t_query))


def phase1_algorithm(t, v, T_val, idx_start, idx_T0):
    t0, t_cut = t[idx_start], t[idx_T0]
    if t0 + 4 * T_val >= t_cut:
        return None
    vs = [interp_v(t, v, t0 + k * T_val) for k in range(5)]
    b1, b2, b3, b4 = vs[0]-vs[1], vs[1]-vs[2], vs[2]-vs[3], vs[3]-vs[4]
    A = np.array([[b2, -b1], [b3, -b2]])
    try:
        u1, u2 = np.linalg.solve(A, np.array([b3, b4]))
    except np.linalg.LinAlgError:
        return None
    disc = u1 ** 2 - 4 * u2
    if disc < 0:
        return None
    d1 = (u1 + np.sqrt(disc)) / 2
    d2 = (u1 - np.sqrt(disc)) / 2
    if not (0 < d1 < 1 and 0 < d2 < 1):
        return None
    tau1 = -T_val / np.log(d1)
    tau2 = -T_val / np.log(d2)
    try:
        x1, x2 = np.linalg.solve(np.array([[1, 1], [d1, d2]]), np.array([b1, b2]))
    except np.linalg.LinAlgError:
        return None
    s1 = x1 / (d1 - 1)
    s2 = x2 / (d2 - 1)
    return s1, s2, tau1, tau2


def phase2_algorithm(t, v, I, I1, T_val, T0, idx_T0, idx_end, s1, s2, tau1, tau2):
    a1, a2 = np.exp(-T0 / tau1), np.exp(-T0 / tau2)
    p1, p2 = np.exp(-T_val / tau1), np.exp(-T_val / tau2)
    T0_t = t[idx_T0]
    t_post = [T0_t + k * T_val for k in range(3)]
    if t_post[-1] > t[idx_end - 1]:
        return None
    vp = [interp_v(t, v, tp) for tp in t_post]
    delta1, delta2 = vp[0]-vp[1], vp[1]-vp[2]
    A3 = np.array([[p1-1, p2-1], [p1*(p1-1), p2*(p2-1)]])
    try:
        y1, y2 = np.linalg.solve(A3, np.array([delta1, delta2]))
    except np.linalg.LinAlgError:
        return None
    dI = I - I1
    if abs(dI) < 1e-6:
        return None
    R1 = (y1 - s1 * a1) / dI
    R2 = (y2 - s2 * a2) / dI
    if R1 <= 0 or R2 <= 0:
        return None
    C1, C2 = tau1 / R1, tau2 / R2
    v10, v20 = s1 + R1 * I, s2 + R2 * I
    return R1, R2, C1, C2, v10, v20


def reconstruct_voltage(t_arr, v, E, R0, R1, R2, C1, C2, v10, v20, I, I1, T0, idx_start):
    tau1, tau2 = R1 * C1, R2 * C2
    t_start = t_arr[idx_start]
    v_rec = v.copy()
    for k in range(idx_start, len(t_arr)):
        tp = t_arr[k] - t_start
        if tp <= T0:
            v_rec[k] = (E - (R0+R1+R2)*I - (v10-R1*I)*np.exp(-tp/tau1) - (v20-R2*I)*np.exp(-tp/tau2))
        else:
            tpp = tp - T0
            v1_T0 = (v10-R1*I)*np.exp(-T0/tau1) + R1*I
            v2_T0 = (v20-R2*I)*np.exp(-T0/tau2) + R2*I
            v_rec[k] = (E - (R0+R1+R2)*I1 - (v1_T0-R1*I1)*np.exp(-tpp/tau1) - (v2_T0-R2*I1)*np.exp(-tpp/tau2))
    return v_rec


def sweep_T_for_cycle(t, v, I, I1, R0, T0, idx_start, idx_T0, idx_end, idx_rest_ref):
    T_max_safe = min(T_MAX, T0 / 4.0)
    T_values = np.linspace(T_MIN, T_max_safe, T_STEPS)
    best_rmse, best_params, best_T = np.inf, None, None
    mask = (t >= t[idx_start]) & (np.arange(len(t)) < idx_end)
    for T_val in T_values:
        r1 = phase1_algorithm(t, v, T_val, idx_start, idx_T0)
        if r1 is None:
            continue
        s1, s2, tau1, tau2 = r1
        r2 = phase2_algorithm(t, v, I, I1, T_val, T0, idx_T0, idx_end, s1, s2, tau1, tau2)
        if r2 is None:
            continue
        R1, R2, C1, C2, v10, v20 = r2
        v_0minus = v[idx_rest_ref]
        E = v_0minus + v10 + v20
        v_rec = reconstruct_voltage(t, v, E, R0, R1, R2, C1, C2, v10, v20, I, I1, T0, idx_start)
        rmse = float(np.sqrt(np.mean((v[mask] - v_rec[mask]) ** 2)))
        if rmse < best_rmse:
            best_rmse, best_T = rmse, T_val
            best_params = (R1, R2, C1, C2, v10, v20, E)
    return best_T, best_rmse, best_params


def main():
    t, v, i, ah = load_data(FILE_PATH)
    cycles = find_cycles(i, ah)

    print(f"\nFound {len(cycles)} candidate cycles in the recording.\n")

    rows = []
    for c in cycles:
        if not c['valid']:
            print(f"  Cycle {c['cycle']:2d}: SKIPPED -- {c['reject_reason']}")
            continue

        idx_start, idx_T0, idx_end = c['idx_start'], c['idx_T0'], c['idx_end']
        idx_rest_ref = c['idx_rest_ref']
        T0 = t[idx_T0] - t[idx_start]
        I = float(np.mean(i[idx_start:idx_T0]))
        I1 = float(np.mean(i[idx_T0:idx_end]))
        R0 = extract_R0(v, i, idx_start, idx_rest_ref)

        best_T, best_rmse, best_params = sweep_T_for_cycle(t, v, I, I1, R0, T0, idx_start, idx_T0, idx_end, idx_rest_ref)

        ah_pre = c['AH_pre']
        soc = 100.0 * (1.0 - ah_pre / Q_REF)

        if best_params is None:
            print(f"  Cycle {c['cycle']:2d}: SKIPPED -- parameter extraction failed for all T")
            continue

        R1, R2, C1, C2, v10, v20, E = best_params
        tau1, tau2 = R1 * C1, R2 * C2
        print(f"  Cycle {c['cycle']:2d}: OK  SOC={soc:6.2f}%  I={I:5.2f}A I1={I1:5.2f}A  "
              f"best_T={best_T:5.2f}s  RMSE={best_rmse*1000:6.2f}mV"
              + (f"  (stepping pulse {c['stepping_I']:.1f}A excluded from fit)" if not np.isnan(c['stepping_I']) else ""))

        rows.append(dict(
            cycle=c['cycle'], SOC_pct=soc, AH_pre=ah_pre, I_A=I, I1_A=I1,
            R0_mOhm=R0*1e3, R1_mOhm=R1*1e3, R2_mOhm=R2*1e3, C1_F=C1, C2_F=C2,
            tau1_s=tau1, tau2_s=tau2, v10_mV=v10*1e3, v20_mV=v20*1e3, E_V=E,
            best_T_s=best_T, RMSE_mV=best_rmse*1e3,
        ))

    results_df = pd.DataFrame(rows).sort_values('SOC_pct', ascending=False).reset_index(drop=True)

    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    out_excel = os.path.join(RESULTS_FOLDER, "battery_param_vs_SOC_Param_0A_10A_30A_Jun30.xlsx")
    results_df.to_excel(out_excel, index=False)
    print(f"\nSaved table ({len(results_df)} valid SOC points): {out_excel}")

    os.makedirs(GRAPHS_FOLDER, exist_ok=True)

    # --- Plot 1: resistances vs SOC ---
    fig1, ax1 = plt.subplots(figsize=(9, 6))
    ax1.plot(results_df['SOC_pct'], results_df['R0_mOhm'], 'o-', color=color_extra, label='R0', linewidth=1.5, markersize=5)
    ax1.plot(results_df['SOC_pct'], results_df['R1_mOhm'], 'o-', color=color_v, label='R1', linewidth=1.5, markersize=5)
    ax1.plot(results_df['SOC_pct'], results_df['R2_mOhm'], 'o-', color=color_c, label='R2', linewidth=1.5, markersize=5)
    ax1.set_xlabel("SOC [%]", fontweight='bold')
    ax1.set_ylabel("Resistance [m\u03a9]", fontweight='bold', color=color_v_l)
    ax1.set_title("ECM resistances vs SOC")
    ax1.invert_xaxis()
    ax1.grid(True, which='major', linestyle="-", alpha=0.6, color='gray')
    ax1.grid(True, which='minor', linestyle="--", alpha=0.3, color='gray')
    ax1.legend(frameon=True, shadow=True)
    fig1.tight_layout()
    p1 = os.path.join(GRAPHS_FOLDER, "ECM_resistances_vs_SOC_Param_0A_10A_30A_Jun30.pdf")
    fig1.savefig(p1, format='pdf', bbox_inches='tight')
    print(f"Saved: {p1}")

    # --- Plot 2: capacitances vs SOC (log scale) ---
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    ax2.plot(results_df['SOC_pct'], results_df['C1_F'], 'o-', color=color_c, label='C1', linewidth=1.5, markersize=5)
    ax2.plot(results_df['SOC_pct'], results_df['C2_F'], 'o-', color=color_v, label='C2', linewidth=1.5, markersize=5)
    ax2.set_yscale('log')
    # --- NEW CODE TO ADD MORE Y-AXIS LABELS ---
    # Define specific values where you want ticks to appear
    custom_ticks = [5000, 10000, 20000, 40000, 100000]
    ax2.set_yticks(custom_ticks)
    ax2.set_yticks(custom_ticks, minor=True) # Ensures minor grid aligns if needed

    # Use ScalarFormatter to show them as regular numbers (e.g., 20000 instead of 2x10^4)
    formatter = ticker.ScalarFormatter()
    formatter.set_scientific(False)
    ax2.yaxis.set_major_formatter(formatter)
    # ------------------------------------------

    ax2.set_xlabel("SOC [%]", fontweight='bold')
    ax2.set_ylabel("Capacitance [F] (log scale)", fontweight='bold', color=color_v_l) # Ensure color_v_l is defined
    ax2.set_title("ECM capacitances vs SOC")
    ax2.invert_xaxis()
    ax2.grid(True, which='major', linestyle="-", alpha=0.6, color='gray')
    ax2.grid(True, which='minor', linestyle="--", alpha=0.3, color='gray')
    ax2.legend(frameon=True, shadow=True)
    fig2.tight_layout()

    p2 = os.path.join(GRAPHS_FOLDER, "ECM_capacitances_vs_SOC_Param_0A_10A_30A_Jun30.pdf")
    fig2.savefig(p2, format='pdf', bbox_inches='tight')
    print(f"Saved: {p2}")
    
    # ax2.set_xlabel("SOC [%]", fontweight='bold')
    # ax2.set_ylabel("Capacitance [F] (log scale)", fontweight='bold', color=color_v_l)
    # ax2.set_title("ECM capacitances vs SOC")
    # ax2.invert_xaxis()
    # ax2.grid(True, which='major', linestyle="-", alpha=0.6, color='gray')
    # ax2.grid(True, which='minor', linestyle="--", alpha=0.3, color='gray')
    # ax2.legend(frameon=True, shadow=True)
    # fig2.tight_layout()
    # p2 = os.path.join(GRAPHS_FOLDER, "ECM_capacitances_vs_SOC_Param_0A_10A_30A_Jun30.pdf")
    # fig2.savefig(p2, format='pdf', bbox_inches='tight')
    # print(f"Saved: {p2}")

    # --- Plot 3: E (OCV) vs SOC, as a sanity check against your OCV-SOC curve ---
    fig3, ax3 = plt.subplots(figsize=(9, 6))
    ax3.plot(results_df['SOC_pct'], results_df['E_V'], 'o-', color=color_v, linewidth=1.5, markersize=5)
    ax3.set_xlabel("SOC [%]", fontweight='bold')
    ax3.set_ylabel("E (OCV) [V]", fontweight='bold', color=color_v_l)
    ax3.set_title("Extracted OCV (E) vs SOC -- compare against your OCV-SOC curve")
    ax3.invert_xaxis()
    ax3.grid(True, which='major', linestyle="-", alpha=0.6, color='gray')
    ax3.grid(True, which='minor', linestyle="--", alpha=0.3, color='gray')
    fig3.tight_layout()
    p3 = os.path.join(GRAPHS_FOLDER, "ECM_E_vs_SOC_Param_0A_10A_30A_Jun30.pdf")
    fig3.savefig(p3, format='pdf', bbox_inches='tight')
    print(f"Saved: {p3}")

    plt.show()


if __name__ == "__main__":
    main()

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.interpolate import interp1d
# import os

# # ─────────────────────────────────────────────────
# #  CONFIG
# # ─────────────────────────────────────────────────
# FILE_PATH        = "/Users/majocuellar/Documents/GitHub/Master_Thesis/Codigo/Test Baterias/Raw/Param_0A_10A_30A_Jun26.CSV"
# # FILE_PATH        = "/Users/majocuellar/Documents/GitHub/Master_Thesis/Codigo/Test Baterias/Raw/Param_0A_10A_18A_Jun26.CSV"
# # FILE_PATH        = "/Users/majocuellar/Documents/GitHub/Master_Thesis/Codigo/Test Baterias/Raw/Param_0A_5A_8A_Jun26.CSV"
# GRAPHS_FOLDER    = "/Users/majocuellar/Documents/GitHub/Master_Thesis/Codigo/Test Baterias/Battery_Graphs_Tests"
# RESULTS_FOLDER   = "/Users/majocuellar/Documents/GitHub/Master_Thesis/Codigo/Test Baterias/Additional CSV"

# I_PHASE1_NOMINAL = 10.0    # [A] expected phase 1 current (for sanity-check print)
# I_PHASE2_NOMINAL = 30.0    # [A] expected phase 2 current

# # I_PHASE1_NOMINAL = 10.0
# # I_PHASE2_NOMINAL = 18.0

# # I_PHASE1_NOMINAL = 5.0
# # I_PHASE2_NOMINAL = 8.0

# LOAD_THRESH = 1.0   # [A] current level that separates "rest" from "loaded"

# # T sweep — paper constraint: T must be in (0, T0/4].
# T_MIN   = 0.5
# T_MAX   = 60.0
# T_STEPS = 120

# PLOT = True
# # ─────────────────────────────────────────────────


# def parse_time_col(series):
#     """Convert the instrument's 'H:MM:SS.s' TIME[s] strings to elapsed seconds."""
#     def _parse(s):
#         try:
#             parts = str(s).split(':')
#             return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
#         except Exception:
#             return np.nan

#     t = series.apply(_parse).to_numpy(dtype=float).copy()   # FIX: .copy() — pandas 3.x can return
#                                                               # a read-only array from to_numpy(),
#                                                               # which crashes the back-fill below.
#     first_valid = int(np.where(~np.isnan(t))[0][0])
#     for k in range(first_valid - 1, -1, -1):
#         t[k] = t[k + 1] - 1.0
#     return t


# def load_data(filepath):
#     """
#     Load the HMC8015 CSV (16 metadata rows, then header, then data).
#     Current is loaded as |IAVG[A]| — FIX: your test sessions don't agree on
#     sign convention (some log discharge as negative, some as positive).
#     Since these are all pure-discharge tests, taking the absolute value is
#     robust to either polarity, instead of a hardcoded flip.
#     """
#     print(f"\nLoading: {filepath}")
#     all_cols = ['TIME[s]', 'UAVG[V]', 'P[W]', 'IAVG[A]',
#                 'WH[Wh]', 'AH[Ah]', 'WHP[Wh]', 'AHP[Ah]',
#                 'WHM[Wh]', 'AHM[Ah]', 'Timestamp']

#     df = pd.read_csv(filepath, sep=';', skiprows=16, names=all_cols,
#                      encoding='utf-8', on_bad_lines='skip')

#     for col in ['UAVG[V]', 'IAVG[A]']:
#         df[col] = pd.to_numeric(
#             df[col].astype(str).str.replace(',', '.'), errors='coerce')

#     df = df.dropna(subset=['UAVG[V]', 'IAVG[A]']).reset_index(drop=True)
#     print(f"  Rows after cleaning: {len(df)}")

#     t = parse_time_col(df['TIME[s]'])
#     v = df['UAVG[V]'].to_numpy(dtype=float)
#     i = np.abs(df['IAVG[A]'].to_numpy(dtype=float))   # FIX: abs() instead of hardcoded -1 flip

#     print(f"  Time   : {t[0]:.1f} s  →  {t[-1]:.1f} s")
#     print(f"  Voltage: {v.min():.3f} V  –  {v.max():.3f} V")
#     print(f"  Current: {i.min():.3f} A  –  {i.max():.3f} A")
#     return t, v, i


# def find_load_start(i, thresh=LOAD_THRESH):
#     """First sample where current rises above rest."""
#     idx = np.where(i > thresh)[0]
#     if len(idx) == 0:
#         raise ValueError("No load current detected. Check threshold or data.")
#     return int(idx[0])


# def find_load_off(i, idx_start, thresh=LOAD_THRESH):
#     """
#     First sample AFTER idx_start where current drops back to rest.
#     FIX (new): this is computed BEFORE searching for the I->I1 step, so the
#     step-detection below can be bounded to the loaded region only.
#     """
#     rel = np.where(i[idx_start:] < thresh)[0]
#     return int(rel[0]) + idx_start if len(rel) > 0 else len(i)


# def find_phase2_start(t, i, idx_start, idx_end):
#     """
#     Largest current jump, searched ONLY inside [idx_start, idx_end).
#     FIX: previously this searched all the way to the end of the file, which
#     meant the load-OFF drop (often the single biggest jump in the whole
#     recording) got mistaken for the I->I1 step. Bounding the search to the
#     loaded region fixes this.
#     """
#     window = i[idx_start + 5: idx_end]      # skip the initial 0->I edge
#     if len(window) < 2:
#         raise ValueError("Loaded region is too short to contain a second "
#                           "current step — this file may be a single-level "
#                           "test, which this method can't process.")
#     di = np.abs(np.diff(window))
#     idx_T0 = int(np.argmax(di)) + 1 + idx_start + 5

#     T0 = t[idx_T0] - t[idx_start]
#     I  = float(np.mean(i[idx_start: idx_T0]))
#     I1 = float(np.mean(i[idx_T0: idx_end]))     # FIX: bounded by idx_end, not open-ended to EOF

#     print(f"\nStep detection:")
#     print(f"  Load-on   : index {idx_start},  t = {t[idx_start]:.1f} s")
#     print(f"  Phase 2   : index {idx_T0},  t = {t[idx_T0]:.1f} s")
#     print(f"  Load-off  : index {idx_end},  t = {t[idx_end] if idx_end < len(t) else float('nan'):.1f} s")
#     print(f"  T0 (phase 1 duration) = {T0:.1f} s")
#     print(f"  I  (phase 1 mean) = {I:.4f} A  (expected ~{I_PHASE1_NOMINAL} A)")
#     print(f"  I1 (phase 2 mean) = {I1:.4f} A  (expected ~{I_PHASE2_NOMINAL} A)")

#     return I, I1, T0, idx_T0


# def extract_R0(v, i, idx_start):
#     """R0 = |ΔV| / |ΔI| at the 0→I step."""
#     if idx_start == 0:
#         raise ValueError("No pre-load sample available for R0 extraction.")
#     v_before  = v[idx_start - 1]
#     v_after   = v[idx_start]
#     I_applied = i[idx_start]
#     R0 = abs(v_before - v_after) / abs(I_applied)
#     print(f"\nR0 = {R0 * 1e3:.4f} mΩ  "
#           f"(ΔV = {(v_before - v_after) * 1e3:.2f} mV,  I = {I_applied:.3f} A)")
#     return R0


# def interp_v(t_data, v_data, t_query):
#     f = interp1d(t_data, v_data, kind='linear', fill_value='extrapolate')
#     return float(f(t_query))


# def phase1_algorithm(t, v, T_val, idx_start, idx_T0):
#     """Extract s1, s2, tau1, tau2 from 5 samples in phase 1."""
#     t0 = t[idx_start]
#     t_cut = t[idx_T0]

#     if t0 + 4 * T_val >= t_cut:
#         return None

#     vs = [interp_v(t, v, t0 + k * T_val) for k in range(5)]

#     b1 = vs[0] - vs[1]
#     b2 = vs[1] - vs[2]
#     b3 = vs[2] - vs[3]
#     b4 = vs[3] - vs[4]

#     A = np.array([[b2, -b1], [b3, -b2]])
#     try:
#         u1, u2 = np.linalg.solve(A, np.array([b3, b4]))
#     except np.linalg.LinAlgError:
#         return None

#     disc = u1 ** 2 - 4 * u2
#     if disc < 0:
#         return None

#     d1 = (u1 + np.sqrt(disc)) / 2
#     d2 = (u1 - np.sqrt(disc)) / 2
#     if not (0 < d1 < 1 and 0 < d2 < 1):
#         return None

#     tau1 = -T_val / np.log(d1)
#     tau2 = -T_val / np.log(d2)

#     try:
#         x1, x2 = np.linalg.solve(
#             np.array([[1, 1], [d1, d2]]),
#             np.array([b1, b2]))
#     except np.linalg.LinAlgError:
#         return None

#     s1 = x1 / (d1 - 1)
#     s2 = x2 / (d2 - 1)
#     return s1, s2, tau1, tau2


# def phase2_algorithm(t, v, I, I1, T_val, T0, idx_T0, idx_end,
#                      s1, s2, tau1, tau2):
#     """Extract R1, R2, C1, C2, v10, v20 from 3 post-step samples."""
#     a1 = np.exp(-T0 / tau1)
#     a2 = np.exp(-T0 / tau2)
#     p1 = np.exp(-T_val / tau1)
#     p2 = np.exp(-T_val / tau2)

#     T0_t = t[idx_T0]
#     t_post = [T0_t + k * T_val for k in range(3)]

#     if t_post[-1] > t[idx_end - 1]:
#         return None

#     vp = [interp_v(t, v, tp) for tp in t_post]
#     delta1 = vp[0] - vp[1]
#     delta2 = vp[1] - vp[2]

#     A3 = np.array([[p1 - 1,        p2 - 1],
#                    [p1 * (p1 - 1), p2 * (p2 - 1)]])
#     try:
#         y1, y2 = np.linalg.solve(A3, np.array([delta1, delta2]))
#     except np.linalg.LinAlgError:
#         return None

#     dI = I - I1
#     if abs(dI) < 1e-6:
#         return None

#     R1 = (y1 - s1 * a1) / dI
#     R2 = (y2 - s2 * a2) / dI
#     if R1 <= 0 or R2 <= 0:
#         return None

#     C1  = tau1 / R1
#     C2  = tau2 / R2
#     v10 = s1 + R1 * I
#     v20 = s2 + R2 * I
#     return R1, R2, C1, C2, v10, v20


# def reconstruct_voltage(t_arr, v, E, R0, R1, R2, C1, C2,
#                         v10, v20, I, I1, T0, idx_start):
#     """Reconstruct terminal voltage from the 2nd-order Thevenin model."""
#     tau1   = R1 * C1
#     tau2   = R2 * C2
#     t_start = t_arr[idx_start]
#     v_rec  = v.copy()

#     for k in range(idx_start, len(t_arr)):
#         tp = t_arr[k] - t_start

#         if tp <= T0:
#             v_rec[k] = (E
#                         - (R0 + R1 + R2) * I
#                         - (v10 - R1 * I) * np.exp(-tp / tau1)
#                         - (v20 - R2 * I) * np.exp(-tp / tau2))
#         else:
#             tpp   = tp - T0
#             v1_T0 = (v10 - R1 * I) * np.exp(-T0 / tau1) + R1 * I
#             v2_T0 = (v20 - R2 * I) * np.exp(-T0 / tau2) + R2 * I
#             v_rec[k] = (E
#                         - (R0 + R1 + R2) * I1
#                         - (v1_T0 - R1 * I1) * np.exp(-tpp / tau1)
#                         - (v2_T0 - R2 * I1) * np.exp(-tpp / tau2))
#     return v_rec


# def sweep_T(t, v, i, I, I1, R0, T0, idx_start, idx_T0, idx_end):
#     T_max_safe = min(T_MAX, T0 / 4.0)
#     T_values   = np.linspace(T_MIN, T_max_safe, T_STEPS)

#     best_rmse   = np.inf
#     best_params = None
#     best_T      = None
#     rmse_curve  = []

#     print(f"\nSweeping T: {T_MIN:.1f} – {T_max_safe:.1f} s  ({T_STEPS} steps)…")

#     mask = (t >= t[idx_start]) & (np.arange(len(t)) < idx_end)

#     for T_val in T_values:
#         r1 = phase1_algorithm(t, v, T_val, idx_start, idx_T0)
#         if r1 is None:
#             rmse_curve.append(np.nan)
#             continue
#         s1, s2, tau1, tau2 = r1

#         r2 = phase2_algorithm(t, v, I, I1, T_val, T0,
#                               idx_T0, idx_end, s1, s2, tau1, tau2)
#         if r2 is None:
#             rmse_curve.append(np.nan)
#             continue
#         R1, R2, C1, C2, v10, v20 = r2

#         v_0minus = v[idx_start - 1]
#         E = v_0minus + v10 + v20

#         v_rec = reconstruct_voltage(t, v, E, R0, R1, R2, C1, C2,
#                                     v10, v20, I, I1, T0, idx_start)

#         rmse = float(np.sqrt(np.mean((v[mask] - v_rec[mask]) ** 2)))
#         rmse_curve.append(rmse)

#         if rmse < best_rmse:
#             best_rmse   = rmse
#             best_T      = T_val
#             best_params = (R1, R2, C1, C2, v10, v20, E, s1, s2, tau1, tau2)

#     return best_T, best_rmse, best_params, T_values, np.array(rmse_curve)


# def print_results(R0, R1, R2, C1, C2, v10, v20, E, tau1, tau2,
#                   best_T, best_rmse):
#     print("\n" + "=" * 57)
#     print("  2nd-ORDER THEVENIN ECM — IDENTIFIED PARAMETERS")
#     print("=" * 57)
#     print(f"  E   (OCV at this SOC)  = {E:.6f}  V")
#     print(f"  R0  (ohmic)            = {R0 * 1e3:.4f}  mΩ")
#     print(f"  R1  (RC pair 1)        = {R1 * 1e3:.4f}  mΩ")
#     print(f"  C1  (RC pair 1)        = {C1:.4f}  F")
#     print(f"  τ1 = R1·C1             = {tau1:.4f}  s")
#     print(f"  R2  (RC pair 2)        = {R2 * 1e3:.4f}  mΩ")
#     print(f"  C2  (RC pair 2)        = {C2:.4f}  F")
#     print(f"  τ2 = R2·C2             = {tau2:.4f}  s")
#     print(f"  v10 (initial V on C1)  = {v10 * 1e3:.4f}  mV")
#     print(f"  v20 (initial V on C2)  = {v20 * 1e3:.4f}  mV")
#     print("-" * 57)
#     print(f"  Best T                 = {best_T:.3f}  s")
#     print(f"  RMSE                   = {best_rmse * 1e3:.4f}  mV")
#     print("  Note: RC pair labels (1/2) are interchangeable —")
#     print("        the algorithm doesn't distinguish fast vs slow.")
#     print("=" * 57)


# def plot_results(t, v, v_rec, i, I, I1, T0, idx_start, idx_end,
#                  T_values, rmse_curve, R0, R1, R2, C1, C2, E,
#                  graphs_folder, tag):
#     os.makedirs(graphs_folder, exist_ok=True)

#     t_start = t[idx_start]
#     t_rel   = t - t_start

#     fig, axes = plt.subplots(3, 1, figsize=(12, 10))
#     fig.suptitle("Battery Parameter Identification — Algebraic Method\n"
#                  "Devarakonda & Hu (2014)", fontsize=13, fontweight='bold')

#     ax = axes[0]
#     ax.plot(t_rel, i, color='#E67E22', linewidth=1.5, label='Measured current (A)')
#     ax.axvline(T0, color='gray', linestyle='--', alpha=0.7, label=f'T₀ = {T0:.0f} s')
#     ax.axvline(t[idx_end] - t_start, color='red', linestyle=':', alpha=0.6, label='Load off')
#     ax.set_ylabel("Current (A)"); ax.set_xlabel("Time from load-on (s)")
#     ax.set_title("Applied Current Profile"); ax.legend(); ax.grid(True, alpha=0.3)

#     ax = axes[1]
#     mask = np.arange(len(t)) >= idx_start
#     ax.plot(t_rel[mask], v[mask],     color='#2980B9', linewidth=1.5, label='Measured', alpha=0.85)
#     ax.plot(t_rel[mask], v_rec[mask], color='#E74C3C', linewidth=1.5, linestyle='--', label='Model')
#     ax.axvline(T0, color='gray', linestyle='--', alpha=0.7)
#     ax.axvline(t[idx_end] - t_start, color='red', linestyle=':', alpha=0.6)
#     ax.set_ylabel("Voltage (V)"); ax.set_xlabel("Time from load-on (s)")
#     ax.set_title("Terminal Voltage: Measured vs Model")
#     param_text = (f"E = {E:.4f} V        R0 = {R0*1e3:.3f} mΩ\n"
#                   f"R1 = {R1*1e3:.3f} mΩ    C1 = {C1:.1f} F    τ1 = {R1*C1:.1f} s\n"
#                   f"R2 = {R2*1e3:.3f} mΩ    C2 = {C2:.1f} F    τ2 = {R2*C2:.1f} s")
#     ax.text(0.02, 0.05, param_text, transform=ax.transAxes, fontsize=8,
#             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
#     ax.legend(); ax.grid(True, alpha=0.3)

#     ax = axes[2]
#     valid = ~np.isnan(rmse_curve)
#     ax.plot(T_values[valid], rmse_curve[valid] * 1e3, color='#27AE60', linewidth=1.5)
#     best_idx = int(np.nanargmin(rmse_curve))
#     ax.axvline(T_values[best_idx], color='red', linestyle='--', label=f'Best T = {T_values[best_idx]:.2f} s')
#     ax.set_ylabel("RMSE (mV)"); ax.set_xlabel("T — sampling interval (s)")
#     ax.set_title("RMSE vs T sweep"); ax.legend(); ax.grid(True, alpha=0.3)

#     plt.tight_layout()
#     out = os.path.join(graphs_folder, f"battery_param_results_{tag}.pdf")
#     fig.savefig(out, format='pdf', bbox_inches='tight')
#     print(f"\nPlot saved: {out}")
#     plt.show()


# def main():
#     print("=" * 57)
#     print("  BATTERY PARAMETER IDENTIFICATION — ALGEBRAIC METHOD")
#     print("  Welion SHP270-30 NMC+ / 2nd-order Thevenin ECM")
#     print("=" * 57)

#     t, v, i = load_data(FILE_PATH)

#     idx_start = find_load_start(i)
#     idx_end   = find_load_off(i, idx_start)
#     I, I1, T0, idx_T0 = find_phase2_start(t, i, idx_start, idx_end)

#     R0 = extract_R0(v, i, idx_start)

#     best_T, best_rmse, best_params, T_values, rmse_curve = sweep_T(
#         t, v, i, I, I1, R0, T0, idx_start, idx_T0, idx_end)

#     if best_params is None:
#         print("\nParameter extraction failed for all T values.")
#         print("Try widening T_MIN/T_MAX or checking signal quality.")
#         return

#     R1, R2, C1, C2, v10, v20, E, s1, s2, tau1, tau2 = best_params
#     print_results(R0, R1, R2, C1, C2, v10, v20, E, tau1, tau2,
#                   best_T, best_rmse)

#     tag = os.path.splitext(os.path.basename(FILE_PATH))[0]

#     os.makedirs(RESULTS_FOLDER, exist_ok=True)
#     results_df = pd.DataFrame({
#         "Parameter": ["E (V)", "R0 (Ohm)", "R1 (Ohm)", "C1 (F)", "tau1 (s)",
#                       "R2 (Ohm)", "C2 (F)", "tau2 (s)",
#                       "v10 (V)", "v20 (V)", "Best_T (s)", "RMSE (V)"],
#         "Value": [E, R0, R1, C1, tau1, R2, C2, tau2,
#                   v10, v20, best_T, best_rmse]
#     })
#     out_excel = os.path.join(RESULTS_FOLDER, f"battery_param_results_{tag}.xlsx")
#     results_df.to_excel(out_excel, index=False)
#     print(f"Results saved: {out_excel}")

#     if PLOT:
#         v_rec = reconstruct_voltage(t, v, E, R0, R1, R2, C1, C2,
#                                     v10, v20, I, I1, T0, idx_start)
#         plot_results(t, v, v_rec, i, I, I1, T0, idx_start, idx_end,
#                      T_values, rmse_curve, R0, R1, R2, C1, C2, E,
#                      GRAPHS_FOLDER, tag)


# if __name__ == "__main__":
#     main()
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.interpolate import interp1d

# # ─────────────────────────────────────────────────
# #  CONFIG — edit these for your Excel file
# # ─────────────────────────────────────────────────
# EXCEL_FILE     = "/Users/majocuellar/Documents/GitHub/Master_Thesis/Codigo/Test Baterias/Raw/battery_test.CSV"   # path to your Excel file
# COL_TIME       = "Time"                # column name for time [s]
# COL_VOLTAGE    = "Voltage"             # column name for terminal voltage [V]
# COL_CURRENT    = "Current"             # column name for current [A] (positive = discharge)
# SHEET_NAME     = 0                     # 0 = first sheet, or write the sheet name

# # Sweep range for T (sampling interval for the algorithm)
# T_MIN          = 0.5                   # seconds
# T_MAX          = 30.0                  # seconds
# T_STEPS        = 60                    # number of T values to try

# # Optional: override automatic step detection
# # Set to None to detect automatically, or set a time value in seconds
# T0_OVERRIDE    = None                  # e.g. 300.0

# # Plot results?
# PLOT           = True
# # ─────────────────────────────────────────────────


# def load_data(filepath, col_time, col_voltage, col_current, sheet):
#     """Load Excel data and return numpy arrays."""
#     print(f"\n📂  Loading: {filepath}")
#     df = pd.read_csv(filepath, sheet_name=sheet)
#     print(f"    Columns found: {list(df.columns)}")
#     print(f"    Rows: {len(df)}")
#     t = df[col_time].to_numpy(dtype=float)
#     v = df[col_voltage].to_numpy(dtype=float)
#     i = df[col_current].to_numpy(dtype=float)
#     return t, v, i


# def detect_current_step(t, i):
#     """
#     Detect the moment the current switches from I to I1.
#     Returns: I, I1, T0 (the switch time), idx_T0 (index in array).
#     Uses the largest change in current as the step location.
#     """
#     di = np.abs(np.diff(i))
#     idx_step = np.argmax(di) + 1          # index of first sample after step
#     T0 = t[idx_step]

#     # Phase 1 average current (before step)
#     I  = np.mean(i[:idx_step])
#     # Phase 2 average current (after step)
#     I1 = np.mean(i[idx_step:])

#     print(f"\n🔍  Current step detected:")
#     print(f"    Phase 1 (discharge):  I  = {I:.4f} A")
#     print(f"    Phase 2 (discharge):  I1 = {I1:.4f} A")
#     print(f"    Step at T0 = {T0:.2f} s  (index {idx_step})")
#     return float(I), float(I1), float(T0), idx_step


# def get_voltage_at(t_data, v_data, t_query):
#     """Interpolate voltage at an arbitrary time t_query."""
#     f = interp1d(t_data, v_data, kind='linear', fill_value='extrapolate')
#     return float(f(t_query))


# def extract_R0(t, v, i, idx_step):
#     """
#     R0 = instant voltage drop / current step magnitude.
#     Uses the sample just before and just after the initial application of load.
#     Assumes t=0 is the start of the test (load just connected).
#     """
#     # Find the first point where current is non-zero
#     idx_load = np.where(np.abs(i) > 0.01)[0]
#     if len(idx_load) == 0:
#         raise ValueError("No current detected. Check current column sign/scale.")
#     idx_start = idx_load[0]

#     v_before = v[idx_start - 1] if idx_start > 0 else v[0]
#     v_after  = v[idx_start]
#     I_applied = i[idx_start]

#     if abs(I_applied) < 1e-6:
#         print("⚠️  Warning: Could not detect clean load application. R0 may be inaccurate.")
#         return 0.0

#     R0 = abs(v_before - v_after) / abs(I_applied)
#     print(f"\n⚡  R0 = {R0*1000:.4f} mΩ  (ΔV = {(v_before-v_after)*1000:.2f} mV, I = {I_applied:.3f} A)")
#     return R0


# def run_phase1_algorithm(t, v, I, T_val, idx_start, idx_T0):
#     """
#     Phase 1 algebraic algorithm.
#     Returns: s1, s2, tau1, tau2 (or None if failed).
#     """
#     # Sample 5 points: v(0+), v(T), v(2T), v(3T), v(4T)
#     t0_plus = t[idx_start]
#     t_samples = [t0_plus + k * T_val for k in range(5)]

#     # Check we don't go past T0
#     t_cutoff = t[idx_T0]
#     if t_samples[-1] >= t_cutoff:
#         return None

#     # Interpolate voltages at those times
#     vs = [get_voltage_at(t, v, ts) for ts in t_samples]

#     # Compute differences b1..b4
#     b = [vs[k] - vs[k+1] for k in range(4)]
#     b1, b2, b3, b4 = b

#     # Step 4a: solve for u1, u2
#     # [b2  -b1] [u1]   [b3]
#     # [b3  -b2] [u2] = [b4]
#     A_mat = np.array([[b2, -b1],
#                       [b3, -b2]])
#     rhs   = np.array([b3, b4])

#     try:
#         u = np.linalg.solve(A_mat, rhs)
#     except np.linalg.LinAlgError:
#         return None

#     u1, u2 = u

#     # Step 4b: solve quadratic d^2 - u1*d + u2 = 0
#     discriminant = u1**2 - 4*u2
#     if discriminant < 0:
#         return None   # complex roots — skip this T

#     d1 = (u1 + np.sqrt(discriminant)) / 2
#     d2 = (u1 - np.sqrt(discriminant)) / 2

#     # d values must be in (0,1) for physical time constants
#     if not (0 < d1 < 1 and 0 < d2 < 1):
#         return None

#     # Step 4c: time constants
#     tau1 = -T_val / np.log(d1)
#     tau2 = -T_val / np.log(d2)
#     if tau1 <= 0 or tau2 <= 0:
#         return None

#     # Step 4d: solve for x1, x2
#     # [1   1 ] [x1]   [b1]
#     # [d1  d2] [x2] = [b2]
#     A2  = np.array([[1,  1],
#                     [d1, d2]])
#     rhs2 = np.array([b1, b2])
#     try:
#         x = np.linalg.solve(A2, rhs2)
#     except np.linalg.LinAlgError:
#         return None

#     x1, x2 = x

#     # Step 4e: recover s1, s2
#     s1 = x1 / (d1 - 1)
#     s2 = x2 / (d2 - 1)

#     return s1, s2, tau1, tau2


# def run_phase2_algorithm(t, v, I, I1, T_val, T0, idx_T0, s1, s2, tau1, tau2):
#     """
#     Phase 2: extract R1, R2 using 3 post-step samples.
#     Returns: R1, R2, C1, C2, v10, v20, E (or None if failed).
#     """
#     # Exponential decay factors at T0
#     a1 = np.exp(-T0 / tau1)
#     a2 = np.exp(-T0 / tau2)

#     # Post-step sampling
#     p1 = np.exp(-T_val / tau1)
#     p2 = np.exp(-T_val / tau2)

#     T0_plus = t[idx_T0]
#     t_post = [T0_plus + k * T_val for k in range(3)]

#     # Check we have data
#     if t_post[-1] > t[-1]:
#         return None

#     vp = [get_voltage_at(t, v, tp) for tp in t_post]
#     delta1 = vp[0] - vp[1]
#     delta2 = vp[1] - vp[2]

#     # Solve for y1, y2
#     # [(p1-1)      (p2-1)   ] [y1]   [delta1]
#     # [p1(p1-1)   p2(p2-1) ] [y2] = [delta2]
#     A3  = np.array([[ (p1-1),      (p2-1)    ],
#                     [ p1*(p1-1),   p2*(p2-1) ]])
#     rhs3 = np.array([delta1, delta2])

#     try:
#         y = np.linalg.solve(A3, rhs3)
#     except np.linalg.LinAlgError:
#         return None

#     y1, y2 = y

#     dI = I - I1   # should be negative (step up in magnitude)
#     if abs(dI) < 1e-6:
#         return None

#     R1 = (y1 - s1 * a1) / dI
#     R2 = (y2 - s2 * a2) / dI

#     if R1 <= 0 or R2 <= 0:
#         return None

#     C1 = tau1 / R1
#     C2 = tau2 / R2

#     v10 = s1 + R1 * I
#     v20 = s2 + R2 * I

#     # Find v(0-) — last point before current starts
#     idx_load = np.where(np.abs(v) > 0)[0]
#     v_0minus = v[0]   # pre-load voltage

#     E = v_0minus + v10 + v20

#     return R1, R2, C1, C2, v10, v20, E


# def reconstruct_voltage(t_arr, E, R0, R1, R2, C1, C2, v10, v20, I, I1, T0, t_start):
#     """Reconstruct terminal voltage from the 2nd-order Thevenin model."""
#     tau1 = R1 * C1
#     tau2 = R2 * C2
#     v_rec = np.zeros_like(t_arr)

#     for k, tk in enumerate(t_arr):
#         tt = tk - t_start
#         if tt < 0:
#             v_rec[k] = E - v10 - v20
#         elif tt <= T0:
#             v_rec[k] = (E - R0*I - R1*I - R2*I
#                         - (v10 - R1*I) * np.exp(-tt/tau1)
#                         - (v20 - R2*I) * np.exp(-tt/tau2))
#         else:
#             tt2 = tt - T0
#             # Capacitor voltages at T0
#             v1_T0 = (v10 - R1*I) * np.exp(-T0/tau1) + R1*I
#             v2_T0 = (v20 - R2*I) * np.exp(-T0/tau2) + R2*I
#             v_rec[k] = (E - R0*I1
#                         - (v1_T0 - R1*I1) * np.exp(-tt2/tau1) - R1*I1
#                         - (v2_T0 - R2*I1) * np.exp(-tt2/tau2) - R2*I1)
#     return v_rec


# def compute_rmse(v_meas, v_model):
#     return np.sqrt(np.mean((v_meas - v_model)**2))


# def sweep_T(t, v, i, I, I1, R0, T0, idx_T0, idx_start):
#     """Sweep T values and return the best parameters (minimum RMSE)."""
#     T_values = np.linspace(T_MIN, T_MAX, T_STEPS)
#     best_rmse = np.inf
#     best_params = None
#     best_T = None
#     rmse_curve = []

#     print(f"\n🔁  Sweeping T from {T_MIN} to {T_MAX} s ({T_STEPS} steps)...")

#     for T_val in T_values:
#         result1 = run_phase1_algorithm(t, v, I, T_val, idx_start, idx_T0)
#         if result1 is None:
#             rmse_curve.append(np.nan)
#             continue

#         s1, s2, tau1, tau2 = result1

#         result2 = run_phase2_algorithm(t, v, I, I1, T_val, T0, idx_T0,
#                                        s1, s2, tau1, tau2)
#         if result2 is None:
#             rmse_curve.append(np.nan)
#             continue

#         R1, R2, C1, C2, v10, v20, E = result2

#         # Reconstruct and compute RMSE over phase 1 + phase 2
#         t_start = t[idx_start]
#         v_rec = reconstruct_voltage(t, E, R0, R1, R2, C1, C2, v10, v20,
#                                     I, I1, T0, t_start)
#         mask = t >= t_start
#         rmse = compute_rmse(v[mask], v_rec[mask])
#         rmse_curve.append(rmse)

#         if rmse < best_rmse:
#             best_rmse = rmse
#             best_T = T_val
#             best_params = (R1, R2, C1, C2, v10, v20, E, s1, s2, tau1, tau2)

#     return best_T, best_rmse, best_params, T_values, np.array(rmse_curve)


# def print_results(R0, R1, R2, C1, C2, v10, v20, E, tau1, tau2, best_T, best_rmse):
#     """Print a formatted results table."""
#     print("\n" + "="*55)
#     print("  2nd-ORDER THEVENIN ECM — IDENTIFIED PARAMETERS")
#     print("="*55)
#     print(f"  E   (OCV at this SOC)  = {E:.6f}  V")
#     print(f"  R0  (ohmic resistance) = {R0*1000:.4f}  mΩ")
#     print(f"  R1  (RC pair 1)        = {R1*1000:.4f}  mΩ")
#     print(f"  C1  (RC pair 1)        = {C1:.4f}  F")
#     print(f"  τ1 = R1·C1             = {tau1:.4f}  s")
#     print(f"  R2  (RC pair 2)        = {R2*1000:.4f}  mΩ")
#     print(f"  C2  (RC pair 2)        = {C2:.4f}  F")
#     print(f"  τ2 = R2·C2             = {tau2:.4f}  s")
#     print(f"  v10 (initial cap. V1)  = {v10*1000:.4f}  mV")
#     print(f"  v20 (initial cap. V2)  = {v20*1000:.4f}  mV")
#     print("-"*55)
#     print(f"  Best T (sweep result)  = {best_T:.3f}  s")
#     print(f"  RMSE                   = {best_rmse*1000:.4f}  mV")
#     print("="*55)


# def plot_results(t, v, v_rec, I, I1, T0, t_start, T_values, rmse_curve,
#                  R0, R1, R2, C1, C2, E):
#     fig, axes = plt.subplots(3, 1, figsize=(12, 10))
#     fig.suptitle("Battery Parameter Identification — Algebraic Method\n"
#                  "Devarakonda & Hu (2014)", fontsize=13, fontweight='bold')

#     # --- Plot 1: Current profile ---
#     ax1 = axes[0]
#     ax1.plot(t - t_start, [I if tt <= T0 else I1 for tt in (t - t_start)],
#              color='#E67E22', linewidth=2, label='Applied current')
#     ax1.axvline(T0, color='gray', linestyle='--', alpha=0.7, label=f'T₀ = {T0:.1f} s')
#     ax1.set_ylabel("Current (A)")
#     ax1.set_xlabel("Time (s)")
#     ax1.set_title("Test Current Profile")
#     ax1.legend()
#     ax1.grid(True, alpha=0.3)

#     # --- Plot 2: Voltage — measured vs reconstructed ---
#     ax2 = axes[1]
#     mask = t >= t_start
#     ax2.plot(t[mask] - t_start, v[mask], color='#2980B9', linewidth=1.5,
#              label='Measured voltage', alpha=0.85)
#     ax2.plot(t[mask] - t_start, v_rec[mask], color='#E74C3C', linewidth=1.5,
#              linestyle='--', label='Reconstructed (model)')
#     ax2.axvline(T0, color='gray', linestyle='--', alpha=0.7)
#     ax2.set_ylabel("Voltage (V)")
#     ax2.set_xlabel("Time (s)")
#     ax2.set_title("Terminal Voltage: Measured vs Model")
#     ax2.legend()
#     ax2.grid(True, alpha=0.3)

#     # Annotate parameters on plot
#     param_text = (f"E={E:.4f}V  R0={R0*1000:.3f}mΩ\n"
#                   f"R1={R1*1000:.3f}mΩ  C1={C1:.2f}F\n"
#                   f"R2={R2*1000:.3f}mΩ  C2={C2:.2f}F")
#     ax2.text(0.02, 0.05, param_text, transform=ax2.transAxes,
#              fontsize=8, verticalalignment='bottom',
#              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

#     # --- Plot 3: RMSE sweep ---
#     ax3 = axes[2]
#     valid = ~np.isnan(rmse_curve)
#     ax3.plot(T_values[valid], rmse_curve[valid] * 1000, color='#27AE60', linewidth=1.5)
#     best_idx = np.nanargmin(rmse_curve)
#     ax3.axvline(T_values[best_idx], color='red', linestyle='--',
#                 label=f'Best T = {T_values[best_idx]:.2f} s')
#     ax3.set_ylabel("RMSE (mV)")
#     ax3.set_xlabel("T — sampling interval (s)")
#     ax3.set_title("RMSE vs T sweep — parameter sensitivity")
#     ax3.legend()
#     ax3.grid(True, alpha=0.3)

#     plt.tight_layout()
#     plt.savefig("battery_param_results.png", dpi=150, bbox_inches='tight')
#     print("\n📊  Plot saved: battery_param_results.png")
#     plt.show()


# # ─────────────────────────────────────────────────
# #  MAIN
# # ─────────────────────────────────────────────────
# def main():
#     print("=" * 55)
#     print("  BATTERY PARAMETER IDENTIFICATION — ALGEBRAIC METHOD")
#     print("  Welion SHP270-30 NMC+ / 2nd-order Thevenin ECM")
#     print("=" * 55)

#     # 1. Load data
#     t, v, i = load_data(EXCEL_FILE, COL_TIME, COL_VOLTAGE, COL_CURRENT, SHEET_NAME)

#     # 2. Detect current step
#     if T0_OVERRIDE is not None:
#         idx_T0 = np.searchsorted(t, T0_OVERRIDE)
#         T0 = float(T0_OVERRIDE)
#         I  = float(np.mean(i[:idx_T0]))
#         I1 = float(np.mean(i[idx_T0:]))
#         print(f"\n⚙️  Using manual T0 = {T0} s")
#     else:
#         I, I1, T0, idx_T0 = detect_current_step(t, i)

#     # 3. Find load start (first non-zero current)
#     idx_load = np.where(np.abs(i) > 0.01)[0]
#     if len(idx_load) == 0:
#         raise ValueError("No current detected.")
#     idx_start = idx_load[0]
#     t_start   = t[idx_start]

#     # 4. R0
#     R0 = extract_R0(t, v, i, idx_step=idx_T0)

#     # 5. Sweep T
#     best_T, best_rmse, best_params, T_values, rmse_curve = sweep_T(
#         t, v, i, I, I1, R0, T0, idx_T0, idx_start)

#     if best_params is None:
#         print("\n❌  Parameter extraction failed for all T values.")
#         print("   Try adjusting T_MIN / T_MAX, or check your data.")
#         return

#     R1, R2, C1, C2, v10, v20, E, s1, s2, tau1, tau2 = best_params

#     # 6. Print results
#     print_results(R0, R1, R2, C1, C2, v10, v20, E, tau1, tau2, best_T, best_rmse)

#     # 7. Save to Excel
#     results_df = pd.DataFrame({
#         "Parameter": ["E (V)", "R0 (Ohm)", "R1 (Ohm)", "C1 (F)",
#                       "tau1 (s)", "R2 (Ohm)", "C2 (F)", "tau2 (s)",
#                       "v10 (V)", "v20 (V)", "Best_T (s)", "RMSE (V)"],
#         "Value": [E, R0, R1, C1, tau1, R2, C2, tau2, v10, v20, best_T, best_rmse]
#     })
#     results_df.to_excel("battery_param_results.xlsx", index=False)
#     print("💾  Results saved: battery_param_results.xlsx")

#     # 8. Plot
#     if PLOT:
#         v_rec = reconstruct_voltage(t, E, R0, R1, R2, C1, C2, v10, v20,
#                                     I, I1, T0, t_start)
#         plot_results(t, v, v_rec, I, I1, T0, t_start, T_values, rmse_curve,
#                      R0, R1, R2, C1, C2, E)


# if __name__ == "__main__":
#     main()
