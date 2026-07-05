% build_SOC_dependent_model.m
%
% Lee la tabla de parámetros-vs-SOC (salida del script Python de
% extracción por lotes) y tu curva OCV-SOC, valida cada ciclo del test
% de parámetros comparando su E contra la curva OCV-SOC, y construye
% las tablas indexadas por SOC para iterEKF.m / iterSOH_EKF.m.
%
% R0, R2, C2: se toman de TODOS los ciclos válidos (consistentemente
% bien identificados).
% R1, C1: SOLO de los ciclos donde la E extraída coincide con la curva
% OCV-SOC dentro de DEV_THRESHOLD_MV.

clear; clc;

%% ---- AJUSTES ----
paramTableFile    = 'battery_param_vs_SOC_Param_0A_10A_30A_Jun30.xlsx';
ocvCurveFile      = 'OCV_SOC_table_Pulsos_10A_3A_1A_Jun22.xlsx';
Q_ref_Ah          = 28.298;     % el valor que decidiste usar
DEV_THRESHOLD_MV  = 20;

%% ---- CARGAR RESULTADOS DEL TEST DE PARÁMETROS (todos los ciclos) ----
paramT = readtable(paramTableFile);

%% ---- CARGAR CURVA OCV-SOC ----
ocvT = readtable(ocvCurveFile);
ocvSOC = ocvT.SOC_pct;   % ajusta el nombre de columna a tu Excel real
ocvV   = ocvT.OCV_V;
[ocvSOC, idx] = sort(ocvSOC);
ocvV = ocvV(idx);

%% ---- VALIDAR CADA CICLO ----
E_curve = interp1(ocvSOC, ocvV, paramT.SOC_pct, 'linear', 'extrap');
dev_mV  = (paramT.E_V - E_curve) * 1000;
isGood  = abs(dev_mV) < DEV_THRESHOLD_MV;

fprintf('Ciclos que pasan validación (|dev| < %d mV): %d de %d\n', ...
        DEV_THRESHOLD_MV, sum(isGood), height(paramT));

%% ---- TABLA R0 / R2 / C2 (todos los ciclos) ----
[socAll, ia] = sort(paramT.SOC_pct);
model.SOC_R0 = socAll' / 100;
model.R0_tab = paramT.R0_mOhm(ia)' / 1000;
model.R2_tab = paramT.R2_mOhm(ia)' / 1000;
model.C2_tab = paramT.C2_F(ia)';

%% ---- TABLA R1 / C1 (solo ciclos validados) ----
good = paramT(isGood, :);
[socGood, ig] = sort(good.SOC_pct);
model.SOC_R1 = socGood' / 100;
model.R1_tab = good.R1_mOhm(ig)' / 1000;
model.C1_tab = good.C1_F(ig)';

%% ---- CURVA OCV-SOC ----
model.SOC = ocvSOC' / 100;
model.OCV = ocvV';

%% ---- CAPACIDAD ----
model.Q = Q_ref_Ah * 3600;   % A*s

save('myModel.mat', 'model');
fprintf('myModel.mat guardado: R0/R2/C2 con %d puntos, R1/C1 con %d puntos\n', ...
        numel(model.SOC_R0), numel(model.SOC_R1));