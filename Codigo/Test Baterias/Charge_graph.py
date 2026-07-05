import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

# =============================================================================
# 1. CARGAR EL ARCHIVO (Soporta automáticamente .csv o .xlsx)
# =============================================================================
# file_path = "/Users/majocuellar/Documents/GitHub/Master_Thesis/Codigo/Test Baterias/Raw/Pulsos_10A_3A_1A_Jun22.CSV"
# file_path = "/Users/majocuellar/Documents/GitHub/Master_Thesis/Codigo/Test Baterias/Raw/Param_0A_10A_30A_Jun26.CSV"
# file_path = "/Users/majocuellar/Documents/GitHub/Master_Thesis/Codigo/Test Baterias/Raw/Charge_20A_4.15V_Jun22.CSV"
file_path = "/Users/majocuellar/Documents/GitHub/Master_Thesis/Codigo/Test Baterias/Raw/Charge_20A_4.15V_Jun17.xlsx"
output_folder = "/Users/majocuellar/Documents/GitHub/Master_Thesis/Codigo/Test Baterias/Battery_Graphs_Tests"

# Asegurar que la carpeta existe
os.makedirs(output_folder, exist_ok=True)

# Extraer el nombre del archivo y construir la ruta de destino
excel_base_name = os.path.splitext(os.path.basename(file_path))[0]
file_ext = os.path.splitext(file_path)[1].lower()
pdf_output_path = os.path.join(output_folder, f"Curve_{excel_base_name}.pdf")

print(f"Loading data from {excel_base_name}{file_ext}...")

# Cargar dinámicamente según la extensión
if file_ext == '.csv':
    # SOLUCIÓN: comment='#' le dice a Pandas que ignore la metadata por completo.
    # skip_blank_lines=True evita que los espacios en blanco arruinen la lectura.
    df = pd.read_csv(file_path, sep=';', comment='#', skip_blank_lines=True, encoding='utf-8')
else:
    # Los Excel tienen 18 filas fijas de metadata
    df = pd.read_excel(file_path, sheet_name="Full2", skiprows=18)

# Limpiar espacios en blanco invisibles en los nombres de las columnas
df.columns = df.columns.str.strip()

# =============================================================================
# 2. CONVERTIR DATOS DE FORMA SEGURA
# =============================================================================
# Sistema de seguridad: si no encuentra la columna, nos dice qué encontró realmente
if "UAVG[V]" not in df.columns:
    print("\n¡ERROR! Pandas no encontró la columna 'UAVG[V]'.")
    print("Las columnas que Pandas sí logró leer fueron:", df.columns.tolist())
    raise KeyError("UAVG[V] no fue encontrada. Por favor revisa las columnas impresas arriba.")

df["UAVG[V]"] = pd.to_numeric(df["UAVG[V]"].astype(str).str.replace(',', '.'), errors="coerce")
df["IAVG[A]"] = pd.to_numeric(df["IAVG[A]"].astype(str).str.replace(',', '.'), errors="coerce")

# =============================================================================
# 3. LIMPIAR Y FILTRAR LOS DATOS
# =============================================================================
# Eliminar tiempos estancados o duplicados
df = df.drop_duplicates(subset=["TIME[s]"], keep="first")

# Mantener filas donde el Voltaje > 0 y la Corriente está activa (carga o descarga)
valid_data_mask = (df["UAVG[V]"] > 0) & (df["IAVG[A]"].abs() > 0.1)
df_clean = df[valid_data_mask].copy()

# Crear un eje de tiempo continuo empezando en 0
df_clean = df_clean.reset_index(drop=True)
df_clean["Continuous_Time[s]"] = df_clean.index

# =============================================================================
# 4. CÁLCULO DE MÉTRICAS
# =============================================================================
# Detectar automáticamente si las columnas P/AH están en mayúsculas o minúsculas
power_col = "P[W]" if "P[W]" in df.columns else "P[w]"
ah_col = "AH[Ah]" if "AH[Ah]" in df.columns else "AH[ah]"

# Convertir de forma segura arreglando las comas decimales europeas
df_clean[power_col] = pd.to_numeric(df_clean[power_col].astype(str).str.replace(',', '.'), errors="coerce")
df_clean[ah_col] = pd.to_numeric(df_clean[ah_col].astype(str).str.replace(',', '.'), errors="coerce")

# Calcular energía consumida (Suma de Watts dividida por 3600 seg)
consumed_Energy = df_clean[power_col].abs().sum() / 3600

# Métricas adicionales
total_duration_sec = df_clean["Continuous_Time[s]"].max()
capacity_2 = df_clean[ah_col].abs().max()

# Imprimir las métricas en la terminal
print("\n=== TEST METRICS ===")
print(f"Consumed Energy: {consumed_Energy:.4f} [Wh]")
print(f"Capacity: {capacity_2:.4f} [Ah]")
print(f"Total Time: {total_duration_sec/3600:.4f} hours")
print("====================\n")

# =============================================================================
# 5. GENERAR EL GRÁFICO
# =============================================================================
fig, ax1 = plt.subplots(figsize=(12, 7))

# Colores académicos profundos
color_v = "#82D982" 
color_c = "#8282FF" 
color_v_l = "#061406"
color_c_l = "#00001A"

# --- Eje Y Principal: Voltaje ---
ax1.set_xlabel("Continuous Time (seconds)", fontweight='bold')
ax1.set_ylabel("Voltage (UAVG [V])", color=color_v_l, fontweight='bold')
ax1.plot(
    df_clean["Continuous_Time[s]"],
    df_clean["UAVG[V]"],
    color=color_v,
    label="Voltage (V)",
    linewidth=2.0
)
ax1.tick_params(axis="y", labelcolor=color_v_l)

# Bloquear eje de Voltaje (Ajustado para 4.15V target)
ax1.set_ylim(2.9, 4.3)
ax1.yaxis.set_major_locator(MultipleLocator(0.1))
ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
ax1.grid(True, which='major', linestyle="-", alpha=0.6, color='gray')
ax1.grid(True, which='minor', linestyle="--", alpha=0.3, color='gray')

# --- Eje Y Secundario: Corriente ---
ax2 = ax1.twinx()

# EJE DE CORRIENTE DINÁMICO: ¡Se ajusta a la corriente máxima de tu prueba!
max_current = df_clean["IAVG[A]"].abs().max()
if pd.isna(max_current) or max_current == 0:
    max_current = 20  # Fallback de seguridad

# Multiplicar por 1.2 da espacio libre en la parte superior del gráfico
ax2.set_ylim(0, max_current * 1.2) 

ax2.set_ylabel("Current (IAVG [A])", color=color_c_l, fontweight='bold')
ax2.plot(
    df_clean["Continuous_Time[s]"],
    df_clean["IAVG[A]"].abs(),  # .abs() lo mantiene en positivo incluso si es descarga
    color=color_c,
    label="Current (A) Absolute",
    alpha=0.85
)
ax2.tick_params(axis="y", labelcolor=color_c_l)

# --- Leyendas y ajustes de diseño ---
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines_1 + lines_2,
    labels_1 + labels_2,
    loc="upper right",
    frameon=True,      
    shadow=True        
)

fig.tight_layout()

# Guardar y Mostrar
print(f"Saving graph as vector PDF to: {pdf_output_path}")
plt.savefig(pdf_output_path, format="pdf", bbox_inches="tight")

plt.show()

# # 1. Load the Excel file for the CHARGE test
# file_path = "/Users/majocuellar/Documents/GitHub/Master_Thesis/Codigo/Test Baterias/Charge_20A_4.15V_Jun17.xlsx"
# # file_path = "/Users/majocuellar/Documents/GitHub/Master_Thesis/Codigo/Test Baterias/Charge_June15.xlsx"
# sheet_name = "Full2" # Assuming it is the same sheet name as before

# output_folder = "/Users/majocuellar/Documents/GitHub/Master_Thesis/Codigo/Test Baterias/Battery_Graphs_Tests"

# # Ensure the folder exists (creates it if it's missing)
# os.makedirs(output_folder, exist_ok=True)

# # Extract the base file name and build the full absolute destination path
# excel_base_name = os.path.splitext(os.path.basename(file_path))[0]
# pdf_output_path = os.path.join(output_folder, f"Curve_{excel_base_name}.pdf")

# df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=18)
# df.columns = df.columns.str.strip()

# # 2. Safely Convert Data to Numbers
# df["UAVG[V]"] = pd.to_numeric(df["UAVG[V]"].astype(str).str.replace(',', '.'), errors="coerce")
# df["IAVG[A]"] = pd.to_numeric(df["IAVG[A]"].astype(str).str.replace(',', '.'), errors="coerce")

# # 3. Remove "Stalled" Time (Duplicates)
# df = df.drop_duplicates(subset=["TIME[s]"], keep="first")

# # 4. Clean the "Trash" Data for a CHARGE Test
# # We keep rows where Voltage is NOT 0.
# # We keep Current > 0.1A to filter out dead noise, but LOW ENOUGH to keep the vital CV phase taper.
# # (No need for .abs() or multiplying by -1, as charging current is naturally positive)
# valid_data_mask = (df["UAVG[V]"] > 0) & (df["IAVG[A]"] > 0.1)
# df_clean = df[valid_data_mask].copy()

# # 5. Create a Continuous Time Axis Starting at 0
# df_clean = df_clean.reset_index(drop=True)
# df_clean["Continuous_Time[s]"] = df_clean.index 

# # 5.5 calculations
# # Automatically detect if power/AH columns are upper or lowercase to avoid KeyErrors
# power_col = "P[W]" if "P[W]" in df.columns else "P[w]"
# ah_col = "AH[Ah]" if "AH[Ah]" in df.columns else "AH[ah]"

# # Safely convert columns to numbers by fixing Catalan commas
# df_clean[power_col] = pd.to_numeric(df_clean[power_col].astype(str).str.replace(',', '.'), errors="coerce")
# df_clean[ah_col] = pd.to_numeric(df_clean[ah_col].astype(str).str.replace(',', '.'), errors="coerce")

# # Calculate Consumed Energy in Wh (Sum of Watts divided by 3600 seconds)
# consumed_Energy = df_clean[power_col].abs().sum() / 3600

# # Additional useful metrics for a charge test
# total_duration_sec = df_clean["Continuous_Time[s]"].max()
# capacity_2 = df_clean[ah_col].abs().max()

# # Print the metrics nicely to your terminal
# print("\n=== CHARGE TEST METRICS ===")
# print(f"Consumed Energy: {consumed_Energy:.4f} [Wh]")
# print(f"Capacity: {capacity_2:.4f} [Ah]")
# print(f"Total Charging Time: {total_duration_sec/3600} hours")

# # 6. Generate the Graph
# fig, ax1 = plt.subplots(figsize=(12, 7))

# # Deep academic RGB colors from your previous setup
# color_v = "#82D982" #"#D5F2D5"
# color_c = "#8282FF" #"#D9D9FF"
# color_v_l = "#061406"
# color_c_l = "#00001A"

# # Primary Y-Axis: Voltage
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

# # Lock Voltage Axis (Adjusted for 4.15V target)
# ax1.set_ylim(2.9, 4.3)
# ax1.yaxis.set_major_locator(MultipleLocator(0.1))
# ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
# ax1.grid(True, which='major', linestyle="-", alpha=0.6, color='gray')
# ax1.grid(True, which='minor', linestyle="--", alpha=0.3, color='gray')

# # Secondary Y-Axis: Current
# ax2 = ax1.twinx()  

# # Lock Current Axis (Scaled for a 20A test)
# # Setting limits from 0 to 22 ensures the 20A flatline sits nicely near the top
# ax2.set_ylim(0, 22.0)

# ax2.set_ylabel("Current (IAVG [A])", color=color_c_l, fontweight='bold')
# ax2.plot(
#     df_clean["Continuous_Time[s]"],
#     df_clean["IAVG[A]"],
#     color=color_c,
#     label="Current (A)",
#     alpha=0.85
# )
# ax2.tick_params(axis="y", labelcolor=color_c_l)

# # Add Titles and Layout Adjustments
# lines_1, labels_1 = ax1.get_legend_handles_labels()
# lines_2, labels_2 = ax2.get_legend_handles_labels()

# ax1.legend(
#     lines_1 + lines_2, 
#     labels_1 + labels_2, 
#     loc="upper right", 
#     frameon=True,      # Gives the legend a nice white background box
#     shadow=True        # Adds a clean drop shadow for readability
# )
# # plt.title("Charge Test: Voltage & Current in Time", fontweight='bold', fontsize=14)
# fig.tight_layout()

# plt.savefig(pdf_output_path, format="pdf", bbox_inches="tight")

# # Show the plot
# plt.show()