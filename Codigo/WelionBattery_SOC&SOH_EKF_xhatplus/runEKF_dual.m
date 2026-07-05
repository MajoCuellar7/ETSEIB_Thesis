% runEKF_dual.m
%
% Dual EKF: simultaneous SOC and SOH (R0, Q) estimation.
%
% ── TWO FILTERS IN PARALLEL ─────────────────────────────────────────────
%
%   SOC filter (iterEKF):
%     Estimates  x = [iR1; iR2; z]
%     Now returns xhatMinus (state prior after Step 1a, before Step 2b)
%     and AhatSOC (state Jacobian df/dx from Step 0).
%
%   SOH filter (iterSOH_EKF):
%     Estimates  theta = [R0; Q]  (ONE joint vector)
%     Receives   xhatMinus and AhatSOC from the SOC filter.
%     Uses xhatMinus (not xhat^+) for output prediction — per Plett p.232.
%
% ── CALL ORDER EACH TIMESTEP (Plett p.232) ──────────────────────────────
%
%   1. Update ekfData.model with latest R0, Q from SOH filter
%   2. Run SOC filter → get xhatMinus and AhatSOC
%   3. Run SOH filter using xhatMinus and AhatSOC
%   4. Store results
%
% ── FILES REQUIRED ───────────────────────────────────────────────────────
%   iterEKF.m       — your original, with 3 small modifications:
%                     (a) signature: add xhatMinus, AhatSOC as outputs
%                     (b) after Step 0: AhatSOC = Ahat;
%                     (c) after Step 1a: xhatMinus = xhat;
%   initEKF.m       — unchanged
%   iterSOH_EKF.m   — SOH parameter filter
%   initSOH_EKF.m   — SOH initialisation
%   dOCVfromSOC.m   — unchanged
%   myModel.mat
%   SAMPLE_DATA_12V_130AH_FULLDISCHARGE_TEST.mat

%% Last updated: July 2st 2026

% 1. LOAD DATA AND MODEL

load myModel
% load('VALIDATION_PULSES.mat');
% Q_ref_cc = 28.298 * 3600;
% % % % % % % % % % % % % % % % load('VALIDATION_DISCHARGE.mat');
% Q_ref_cc = 27.6160 * 3600;
% load('VALIDATION_CHARGE.mat');
% Q_ref_cc = 27.6160 * 3600;
load('VALIDATION_Steps');
Q_ref_cc = 28 * 3600;
% load('SAMPLE_DATA_12V_130AH_FULLDISCHARGE_TEST.mat');

deltat  = double(PARAMETERS_GLOBAL_DELTAT);
current = double(MEAS_BATTERY_I_ARRAY(:));   % positive = discharge
voltage = double(MEAS_BATTERY_V_ARRAY(:));
time    = (0 : length(current)-1)' * deltat;
N       = length(voltage);

% 2. SOC EKF TUNING
SigmaX0 = diag([1e-6, 1e-6, 2e-4]);   % initial state uncertainty
SigmaV  = 1e-1;                         % voltage measurement noise [V²]
SigmaW  = 1e-1;                         % process noise

% 3. SOH EKF TUNING
% SigmaTheta0: initial uncertainty on [R0; Q].
%   R0 ~ 0.0048 Ohm, 10% uncertainty → (0.1*R0)² ≈ 2.3e-7 Ohm²
%   Q  ~ 5.06e5 A·s,  5% uncertainty → (0.05*Q)²  ≈ 6.4e8  (A·s)²
%
% SigmaR: how fast parameters are allowed to drift per sample.
%   Keep very small — R0 and Q change over hours/cycles, not seconds.
%   If Q estimate oscillates, reduce SigmaR(2,2) further.
%
% SigmaE: voltage noise for SOH filter — same sensor, same value as SigmaV.

% 4. INITIALISE SOC FILTER

ekfData = initEKF(voltage(1), SigmaX0, SigmaV, SigmaW, model);

% 3. GETTING VALUES FOR PARAMETERS AND INITIALISE FILTER

% R0_nom = model.R0;   % 0.0048 Ohm
% Q_nom  = model.Q;    % 5.06e5 A·s

R0_init = interp1(model.SOC_all, model.R0_tab, ekfData.xhat(ekfData.zkInd), 'linear', 'extrap');

model_for_soh     = model;       % copy all the table fields
model_for_soh.R0  = R0_init;    % add the scalar R0 initSOH_EKF needs

R0_nom = R0_init;
Q_nom  = model.Q;

% SigmaTheta0 = diag([(0.10 * R0_nom)^2, ...   % R0 uncertainty (Ohm²)
%                      (0.05 * Q_nom )^2]);      % Q  uncertainty (A·s)²
SigmaTheta0 = diag([(0.06 * R0_nom)^2, ...   % R0: ±3% initial uncertainty
                     (0.02 * Q_nom )^2]);      % Q:  ±5% initial uncertainty

SigmaR = diag([1e-12, ...    %1e-12% R0 drift per sample (Ohm²)
               1e-1   ]);  %1e2   % Q  drift per sample (A·s)²
                              % Start conservative; increase if filter
                              % is too slow to track changes.

% SigmaE = SigmaV;             % same voltage sensor
SigmaE = 0.1;%1e-3;     % σ = 10 mV — realistic voltage sensor noise
                   % must be smaller than R0×Δi = 0.062 V to see R0

% 4. INITIALISE SOH FILTER

sohData = initSOH_EKF(model_for_soh, SigmaTheta0, SigmaR, SigmaE);

% 5. PRE-ALLOCATE STORAGE

sochat   = zeros(N, 1);
socbound = zeros(N, 1);
R0hat    = zeros(N, 1);
R0bound  = zeros(N, 1);
Qhat     = zeros(N, 1);
Qbound   = zeros(N, 1);

% 6. MAIN DUAL EKF LOOP

hwait = waitbar(0, 'Running Dual EKF...');
WARMUP_SAMPLES = 500; % 500

for k = 1 : N

  vk = voltage(k);
  ik = current(k);

  % ── SOH → SOC coupling ─────────────────────────────────────────────
  % Update the model inside ekfData with the latest R0 and Q from the
  % SOH filter before calling iterEKF. iterEKF reads R0 and Q from
  % ekfData.model at the top of every call, so this is automatic.
  % (At k=1, sohData.thetahat = nominal values — no jump.)
  ekfData.model.R0 = sohData.thetahat(sohData.R0Ind);
  ekfData.model.Q  = sohData.thetahat(sohData.QInd);

  % ── SOC FILTER ─────────────────────────────────────────────────────
  % iterEKF now returns two extra outputs:
  %   xhatMinus : state prior [iR1^-; iR2^-; z^-] after Step 1a
  %   AhatSOC   : 3×3 Ahat = df/dx computed in Step 0
  % Both are needed by the SOH filter below.
  [sochat(k), socbound(k), xhatMinus, AhatSOC, ekfData] = ...
      iterEKF(vk, ik, deltat, ekfData);

  % ── SOH FILTER ─────────────────────────────────────────────────────
  % Receives xhatMinus (NOT xhat^+) for output prediction — per Plett p.232.
  % Receives AhatSOC to update Phi without recomputing Ahat.
  % [thetahat, thetabnd, sohData] = iterSOH_EKF(vk, ik, deltat, xhatMinus, AhatSOC, sohData);
  [thetahat, thetabnd, sohData] = iterSOH_EKF(vk, ik, deltat, xhatMinus, AhatSOC, sohData, k <= WARMUP_SAMPLES);


  % ── STORE RESULTS ───────────────────────────────────────────────────
  R0hat(k)   = thetahat(sohData.R0Ind);
  R0bound(k) = thetabnd(sohData.R0Ind);
  Qhat(k)    = thetahat(sohData.QInd);
  Qbound(k)  = thetabnd(sohData.QInd);

  if mod(k, 1000) == 0
    waitbar(k / N, hwait);
  end

end

close(hwait);

% 7. REFERENCE SOC TRACES

% ── Test reference (coulomb counting with Q degraded 10%) ────────────
% FactorQ = 0.1;
% FactorR = 0.0;
% 
% R0_at_soc1 = interp1(model.SOC_all, model.R0_tab, sochat(1), 'linear', 'extrap');
% 
% OCV_test = voltage(1) + current(1) * (R0_at_soc1 * (1 - FactorR));
% OCV_test = min(max(OCV_test, model.OCV(end)), model.OCV(1));
% soc0_test = interp1(model.OCV, model.SOC, OCV_test, 'linear', 'extrap');
% soc0_test = min(1, max(0, soc0_test));
% 
% soc_test    = zeros(N, 1);
% soc_test(1) = soc0_test;
% for k = 2 : N
%   soc_test(k) = soc_test(k-1) - ...
%       (deltat / (model.Q * (1 - FactorQ))) * current(k-1);
% end
% soc_test = min(1, max(0, soc_test));

% ── Coulomb counting reference (nominal Q) ────────────────────────────
OCV_est = voltage(1) + current(1) * R0_at_soc1;
OCV_est = min(max(OCV_est, model.OCV(end)), model.OCV(1));
soc0    = interp1(model.OCV, model.SOC, OCV_est, 'linear', 'extrap');
soc0    = min(1, max(0, soc0));

soc_ref    = zeros(N, 1);
soc_ref(1) = soc0;
for k = 2 : N
  soc_ref(k) = soc_ref(k-1) - (deltat / Q_ref_cc) * current(k-1);
end
soc_ref = min(1, max(0, soc_ref));

% 8. PLOTS
t_min = time / 60;

% ── Figure 1: SOC estimate ────────────────────────────────────────────
figure(1); clf;
% plot(t_min, 100*soc_test, 'c',  'LineWidth', 1.2); hold on;
plot(t_min, 100*soc_ref,  'g',  'LineWidth', 1.2); hold on
plot(t_min, 100*sochat,   'b',  'LineWidth', 1.5);
plot([t_min; NaN; t_min], [100*(sochat+socbound); NaN; 100*(sochat-socbound)], 'm');
xlabel('Time (min)'); ylabel('SOC (%)');
title('SOC Estimation — Dual EKF');
% label_test = sprintf('Test ref (Q −%.0f%%, R0 −%.0f%%)', FactorQ*100, FactorR*100);
% legend(label_test, 'Coulomb counting', 'EKF estimate', '3\sigma bounds');
legend('Coulomb counting', 'EKF estimate', '3\sigma bounds');
ylim([0 110]); grid on;

% ── Figure 2: SOC error ───────────────────────────────────────────────
figure(2); clf;
% plot(t_min, 100*(soc_test - sochat), 'c', 'LineWidth', 1.2); hold on;
plot(t_min, 100*(soc_ref  - sochat), 'g', 'LineWidth', 1.2); hold on
plot([t_min; NaN; t_min], [100*socbound; NaN; -100*socbound], 'm');
xlabel('Time (min)'); ylabel('SOC error (%)');
title('SOC Error: coulomb counting vs Dual EKF');
% legend('Test difference', 'CC difference', '3\sigma bounds', 'Location', 'northwest');
legend('CC difference', '3\sigma bounds', 'Location', 'northwest');
ylim([-10 10]); grid on;

% --- VALIDATION: compare the SOH filter's online R0 estimate against ---
% --- the as-characterized table value AT THE SOC the cell was at -------
R0_table_at_SOC = interp1(model.SOC_all, model.R0_tab, sochat, 'linear', 'extrap');
SOH_R0 = (R0_table_at_SOC ./ R0hat) * 100;   % was: (model.R0 ./ R0hat)*100
SOH_Q  = (Qhat / model.Q) * 100;
% ── Figure 3: R0 estimate vs its SOC-appropriate nominal ───────────────
figure(3); clf;
plot(time/60, R0hat*1000, 'r', 'LineWidth', 1.5); hold on;
plot(time/60, R0_table_at_SOC*1000, 'k--', 'LineWidth', 1.2);
plot([time/60; NaN; time/60], ...
     [(R0hat+R0bound)*1000; NaN; (R0hat-R0bound)*1000], 'm');
xlabel('Time (min)'); ylabel('R_0 (m\Omega)');
title('R_0: SOH estimate vs as-characterized nominal (at matching SOC)');
legend('R_0 estimate (SOH filter)', 'R_0 table (validation, same SOC)', '3\sigma bounds');
grid on;

% ── Figure 4: Q estimate ──────────────────────────────────────────────
figure(4); clf;
plot(t_min, Qhat/3600, 'b', 'LineWidth', 1.5); hold on;
plot([t_min; NaN; t_min], ...
     [(Qhat+Qbound)/3600; NaN; (Qhat-Qbound)/3600], 'm');
yline(model.Q/3600, 'k--', 'LineWidth', 1.2);
xlabel('Time (min)'); ylabel('Q (Ah)');
title('Capacity Q Estimation — SOH Parameter Filter');
legend('Q estimate', '3\sigma bounds', 'Nominal Q');
grid on;

% ── Figure 5: SOH percentages ─────────────────────────────────────────
% SOH_Q  = (Qhat  / model.Q)  * 100;
% SOH_R0 = (model.R0 ./ R0hat) * 100;

figure(5); clf;
plot(t_min, SOH_Q,  'b', 'LineWidth', 1.5); hold on;
plot(t_min, SOH_R0, 'r', 'LineWidth', 1.5);
yline(100, 'k--', 'LineWidth', 1.0);
xlabel('Time (min)'); ylabel('SOH (%)');
title('State of Health');
legend('SOH_Q (capacity)', 'SOH_{R_0} (resistance)', 'New cell');
ylim([0 120]); grid on;

% 9. PRINTED SUMMARY

rmse_cc   = sqrt(mean((100*(soc_ref  - sochat)).^2));
% rmse_test = sqrt(mean((100*(soc_test - sochat)).^2));

fprintf('\n════════════════════════════════════\n');
fprintf('  DUAL EKF RESULTS\n');
fprintf('════════════════════════════════════\n');
fprintf('SOC RMSE vs coulomb counting:  %.4f %%\n', rmse_cc);
% fprintf('SOC RMSE vs test reference:    %.4f %%\n', rmse_test);
fprintf('\nFinal SOH estimates:\n');
fprintf('  R0 = %.4f mΩ   (nominal %.4f mΩ)   SOH_R0 = %.1f %%\n', ...
        R0hat(end)*1000, R0_init*1000, SOH_R0(end));
fprintf('  Q  = %.2f Ah   (nominal %.2f Ah)    SOH_Q  = %.1f %%\n', ...
        Qhat(end)/3600,  model.Q/3600,  SOH_Q(end));
fprintf('  R0 3σ bound: ±%.4f mΩ\n', R0bound(end)*1000);
fprintf('  Q  3σ bound: ±%.2f Ah\n',  Qbound(end)/3600);
fprintf('════════════════════════════════════\n\n');

% ── Export results for Python plotting ───────────────────────────────
results = table(time, voltage, current, ...
                sochat, socbound, soc_ref, ...
                R0hat, R0bound, R0_table_at_SOC, ...
                Qhat, Qbound, ...
    'VariableNames', {'time_s','voltage_V','current_A', ...
                      'sochat','socbound','soc_ref', ...
                      'R0hat_Ohm','R0bound_Ohm','R0_table_Ohm', ...
                      'Qhat_As','Qbound_As'});

results_path = 'Results/Steps_EKF_dual_results.csv';
writetable(results, results_path);
fprintf('Results saved for plotting: %s\n', results_path);
