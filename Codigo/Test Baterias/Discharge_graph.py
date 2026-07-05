import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

# 1. Load the Excel file
file_path = "/Users/majocuellar/Documents/GitHub/Master_Thesis/Codigo/Test Baterias/Raw/Discharge_4A_3.0V_June16.xlsx"
sheet_name = "Full2" 

output_folder = "/Users/majocuellar/Documents/GitHub/Master_Thesis/Codigo/Test Baterias/Battery_Graphs_Tests"

# Ensure the folder exists (creates it if it's missing)
os.makedirs(output_folder, exist_ok=True)

# Extract the base file name and build the full absolute destination path
excel_base_name = os.path.splitext(os.path.basename(file_path))[0]
pdf_output_path = os.path.join(output_folder, f"Curve_{excel_base_name}.pdf")
# Skip the first 18 metadata rows so the columns align correctly
df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=18)
df.columns = df.columns.str.strip()

# Automatically detect if power/AH columns are upper or lowercase to avoid KeyErrors
power_col = "P[W]" if "P[W]" in df.columns else "P[w]"
ah_col = "AH[Ah]" if "AH[Ah]" in df.columns else "AH[ah]"

# 2. Safely Convert Data to Numbers
df["UAVG[V]"] = pd.to_numeric(df["UAVG[V]"].astype(str).str.replace(',', '.'), errors="coerce")
df["IAVG[A]"] = pd.to_numeric(df["IAVG[A]"].astype(str).str.replace(',', '.'), errors="coerce")

# 3. Remove "Stalled" Time (Duplicates)
df = df.drop_duplicates(subset=["TIME[s]"], keep="first")

# 4. Clean the "Trash" Data
valid_data_mask = (df["UAVG[V]"] != 0) & (df["IAVG[A]"].abs() >= 3.9)
df_clean = df[valid_data_mask].copy()

# 5. Create a Continuous Time Axis Starting at 0
# FIX: Moved this ABOVE the calculations so the column exists when you calculate duration!
df_clean = df_clean.reset_index(drop=True)
df_clean["Continuous_Time[s]"] = df_clean.index 

# 4.5 Use the data to compute things
# FIX: Using the dynamically found column names (P[W] and AH[Ah])
df_clean[power_col] = pd.to_numeric(df_clean[power_col].astype(str).str.replace(',', '.'), errors="coerce")
df_clean[ah_col] = pd.to_numeric(df_clean[ah_col].astype(str).str.replace(',', '.'), errors="coerce")

# Metrics Calculations
total_duration_sec = df_clean["Continuous_Time[s]"].max()
avg_current = df_clean["IAVG[A]"].abs().mean() # Using absolute value for accurate capacity math
num_samples = len(df_clean)

# FIX: Corrected physics formulas for Watt-hours and Amp-hours
discharge_Energy = df_clean[power_col].abs().sum() / 3600
capacity_1 = (avg_current * total_duration_sec) / 3600
capacity_2 = df_clean[ah_col].abs().max() # Cumulative Ah column max shows total capacity out

print("\n=== DISCHARGE TEST METRICS ===")
print(f"Discharge Energy:    {discharge_Energy:.4f} [Wh]")
print(f"Average current: {avg_current} [A]")
print(f"Capacity (Avg current):  {capacity_1:.4f} [Ah]")
print(f"Capacity (Analyzer):   {capacity_2:.4f} [Ah]")
print(f"Total Duration:      {total_duration_sec/3600} hours")
print(f"Number of Samples:   {num_samples}")

# 6. Generate the Graph
fig, ax1 = plt.subplots(figsize=(12, 7)) 

color_v = "#82D982" 
color_c = "#8282FF" 
color_v_l = "#061406"
color_c_l = "#00001A"

# Primary Y-Axis: Voltage
ax1.set_xlabel("Continuous Time (seconds)", fontweight='bold')
ax1.set_ylabel("Voltage (UAVG [V])", color=color_v_l, fontweight='bold')
ax1.plot(
    df_clean["Continuous_Time[s]"],
    df_clean["UAVG[V]"],
    color=color_v,
    label="Voltage (V)",
    linewidth=1.5
)
ax1.tick_params(axis="y", labelcolor=color_v_l)

# Make Voltage Granular
ax1.set_ylim(2.9, 4.3)
ax1.yaxis.set_major_locator(MultipleLocator(0.1))
ax1.yaxis.set_minor_locator(AutoMinorLocator(2))

# Add a detailed grid
ax1.grid(True, which='major', linestyle="-", alpha=0.6, color='gray')
ax1.grid(True, which='minor', linestyle="--", alpha=0.3, color='gray')

# Secondary Y-Axis: Current
ax2 = ax1.twinx()  
ax2.set_ylim(-3.0, -4.0)
ax2.set_ylabel("Current (IAVG [A])", color=color_c_l, fontweight='bold')
ax2.plot(
    df_clean["Continuous_Time[s]"],
    df_clean["IAVG[A]"],
    color=color_c,
    label="Current (A)",
    alpha=0.7 
)
ax2.tick_params(axis="y", labelcolor=color_c_l)

# Add Titles and Layout Adjustments
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines_1 + lines_2, 
    labels_1 + labels_2, 
    loc="upper right", 
    frameon=True,      # Gives the legend a nice white background box
    shadow=True        # Adds a clean drop shadow for readability
)
# plt.title("Discharge Test: Voltage & Current in Time", fontweight='bold', fontsize=14)
fig.tight_layout()

plt.savefig(pdf_output_path, format="pdf", bbox_inches="tight")

plt.show()