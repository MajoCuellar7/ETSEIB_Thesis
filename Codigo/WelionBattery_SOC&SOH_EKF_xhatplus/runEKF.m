% runEKF: Runs an extended Kalman filter for saved E2 dynamic data and
% an E2 cell model.
%
% Copyright (c) 2016 by Gregory L. Plett of 
% University of Colorado Colorado Springs (UCCS). 
%
% This work is licensed under a Creative Commons 
% Attribution-NonCommercial-ShareAlike 4.0 Intl. License, v. 1.0
%
% It is provided "as is", without express or implied warranty, for 
% educational and informational purposes only.
%
% This file is provided as a supplement to: Plett, Gregory L., "Battery
% Management Systems, Volume II, Equivalent-Circuit Methods," Artech House, 
% 2015.

%% Last updated: July 2st 2026
% Load model file corresponding to a cell of this type
load myModel

% Load cell-test data to be used for this batch experiment
% load('VALIDATION_PULSES.mat');
% Q_ref_cc = 28.298 * 3600;
% % % % % % % % % % % % % % % % load('VALIDATION_DISCHARGE.mat');
% Q_ref_cc = 27.6160 * 3600;
% load('VALIDATION_CHARGE.mat');
% Q_ref_cc = 27.6160 * 3600;
load('VALIDATION_Steps');
Q_ref_cc = 28 * 3600;

deltat  = double(PARAMETERS_GLOBAL_DELTAT);
current = double(MEAS_BATTERY_I_ARRAY(:));   % discharge > 0
voltage = double(MEAS_BATTERY_V_ARRAY(:));
time    = (0 : length(current)-1)' * deltat; % build time vector
N = length(voltage);

% % Reserve storage for computed results, for plotting
% sochat = zeros(size(voltage));
% socbound = zeros(size(voltage));

% Covariance values
SigmaX0 = diag([1e-6 1e-6 2e-4]); % [1e-6 1e-8 2e-4] % uncertainty of initial state
SigmaV = 1e-1; %2e-1; % Uncertainty of voltage sensor, output equation
SigmaW = 1e-1; %2e-1; % Uncertainty of current sensor, state equation

% Create ekfData structure and initialize variables using first
% voltage measurement and first temperature measurement
ekfData = initEKF(voltage(1), SigmaX0, SigmaV, SigmaW, model);

ekfData.model.R0 = interp1(model.SOC_all, model.R0_tab, ekfData.xhat(ekfData.zkInd), 'linear', 'extrap');

sochat = zeros(N, 1);
socbound = zeros(N, 1);

hwait = waitbar(0,'Analizando...'); 

for k = 1:length(voltage)
  vk = voltage(k); % "measure" voltage
  ik = current(k); % "measure" current
  
  ekfData.model.R0 = interp1(model.SOC_all, model.R0_tab, ekfData.xhat(ekfData.zkInd), 'linear', 'extrap');
  % Update SOC (and other model states)
  [sochat(k),socbound(k), ~, ~, ekfData] = iterEKF(vk, ik, deltat, ekfData);
  
  % update waitbar periodically, but not too often (slow procedure)
  if mod(k,1000)==0
    waitbar(k/length(current),hwait);
  end
end

close(hwait);

% --- Reference SOC: test (Q degraded) ---
% FactorQ = 0.1;
% FactorR = 0.0;
% R0_init = interp1(model.SOC_all, model.R0_tab, sochat(1), 'linear', 'extrap');
% OCV_test = voltage(1) + current(1) * (R0_init * (1 - FactorR));
% OCV_test = min(max(OCV_test, model.OCV(end)), model.OCV(1));
% soc0_test = min(1, max(0, interp1(model.OCV, model.SOC, OCV_test, 'linear', 'extrap')));
% 
% soc_test = zeros(N, 1);
% soc_test(1) = soc0_test;
% for k = 2 : N
%   soc_test(k) = soc_test(k-1) - (deltat / (model.Q * (1 - FactorQ))) * current(k-1);
% end
% soc_test = min(1, max(0, soc_test));

% --- Reference SOC: coulomb counting ---
OCV_est = voltage(1) + current(1) * R0_init;
OCV_est = min(max(OCV_est, model.OCV(end)), model.OCV(1));
soc0    = min(1, max(0, interp1(model.OCV, model.SOC, OCV_est, 'linear', 'extrap')));

soc_ref = zeros(N, 1);
soc_ref(1) = soc0;
for k = 2 : N
  soc_ref(k) = soc_ref(k-1) - (deltat / Q_ref_cc) * current(k-1);
end
soc_ref = min(1, max(0, soc_ref));

% --- Figures and analysis ---
figure(1); clf;
% plot(time/60, 100*soc_test, 'c', 'LineWidth', 1.2); hold on
plot(time/60, 100*soc_ref,  'g', 'LineWidth', 1.2); hold on
plot(time/60, 100*sochat,   'b', 'LineWidth', 1.5);
plot([time/60; NaN; time/60], ...
     [100*(sochat+socbound); NaN; 100*(sochat-socbound)], 'm');
xlabel('Time (min)'); ylabel('SOC (%)');
% label_test = sprintf('Test ref (Q −%.0f%%, R0 −%.0f%%)', FactorQ*100, FactorR*100);
% legend(label_test, 'Coulomb counting', 'EKF estimate', 'EKF 3σ bounds');
legend('Coulomb counting', 'EKF estimate', 'EKF 3σ bounds');
ylim([0 110]); grid on; title('SOC Estimation — EKF');

figure(2); clf;
% plot(time/60, 100*(soc_test - sochat), 'c', 'LineWidth', 1.2); hold on
plot(time/60, 100*(soc_ref  - sochat), 'g', 'LineWidth', 1.2); hold on
plot([time/60; NaN; time/60], [100*socbound; NaN; -100*socbound], 'm');
xlabel('Time (min)'); ylabel('SOC error (%)');
% legend('Test diff', 'CC diff', '3σ bounds', 'Location', 'northwest');
legend('CC diff', '3σ bounds', 'Location', 'northwest');
ylim([-10 10]); grid on; title('SOC Error');

% rmse_vs_test = sqrt(mean((100*(soc_test - sochat)).^2));
rmse_vs_ref  = sqrt(mean((100*(soc_ref  - sochat)).^2));
% fprintf('\nRMSE vs test ref:  %.4f %%\n', rmse_vs_test);
fprintf('RMSE vs CC:        %.4f %%\n', rmse_vs_ref);

% ── Export results for Python plotting ───────────────────────────────
results = table(time, voltage, current, sochat, socbound, soc_ref, ...
    'VariableNames', {'time_s','voltage_V','current_A','sochat','socbound','soc_ref'});
results_path = 'Results/Steps_EKF_soc_results.csv';
writetable(results, results_path);