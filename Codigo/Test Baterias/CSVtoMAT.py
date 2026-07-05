import pandas as pd
import numpy as np
from scipy.io import savemat
import os

RAW_FOLDER = "/Users/majocuellar/Documents/GitHub/Master_Thesis/Codigo/Test Baterias/Raw"

def csv_to_mat(csv_filename, mat_filename, mode='discharge'):
    csv_path = os.path.join(RAW_FOLDER, csv_filename)
    mat_path = os.path.join(RAW_FOLDER, mat_filename)

    all_cols = ['TIME[s]','UAVG[V]','P[W]','IAVG[A]','WH[Wh]','AH[Ah]',
                'WHP[Wh]','AHP[Ah]','WHM[Wh]','AHM[Ah]','Timestamp']
    df = pd.read_csv(csv_path, sep=';', skiprows=16, names=all_cols,
                     encoding='utf-8', on_bad_lines='skip')
    for col in ['UAVG[V]', 'IAVG[A]']:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',','.'), errors='coerce')
    df = df.dropna(subset=['UAVG[V]','IAVG[A]']).reset_index(drop=True)

    v     = df['UAVG[V]'].to_numpy(dtype=float)
    i_raw = df['IAVG[A]'].to_numpy(dtype=float)

    if mode == 'discharge':
        i = np.abs(i_raw)
    elif mode == 'charge':
        i = -np.abs(i_raw)
    else:
        i = i_raw

    savemat(mat_path, {
        'PARAMETERS_GLOBAL_DELTAT': np.array([[1.0]]),
        'MEAS_BATTERY_I_ARRAY':     i.reshape(-1,1),
        'MEAS_BATTERY_V_ARRAY':     v.reshape(-1,1),
    })
    print(f"Saved {mat_path}: {len(v)} samples, {len(v)/3600:.2f} h")

csv_to_mat("Charge_20A_4.15V_Jun22.CSV",  "VALIDATION_CHARGE.mat",    'charge')
# csv_to_mat("Discharge_4A_3_0V_June16_cleaned.CSV", "VALIDATION_DISCHARGE.mat", 'discharge')
# csv_to_mat("Pulsos_10A_3A_1A_Jun22.CSV",   "VALIDATION_PULSES.mat",    'discharge')
# csv_to_mat("Param_0A_10A_30A_Jun30.CSV",   "VALIDATION_Steps.mat",    'discharge')