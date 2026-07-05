load myModel
load('VALIDATION_DISCHARGE.mat');
% load('VALIDATION_CHARGE.mat');
% load('SAMPLE_DATA_12V_130AH_FULLDISCHARGE_TEST.mat');
% load('VALIDATION_PULSES.mat');

deltat  = double(PARAMETERS_GLOBAL_DELTAT);
current = double(MEAS_BATTERY_I_ARRAY(:));   % positive = discharge
voltage = double(MEAS_BATTERY_V_ARRAY(:));
time    = (0 : length(current)-1)' * deltat;
N       = length(voltage);



figure(1); clf;
plot(time, voltage, 'c',  'LineWidth', 1.2); hold on;
plot(time, current,  'g',  'LineWidth', 1.2);
xlabel('Time (min)'); ylabel('current/voltage');
title('lets see the current and voltage values');
legend('Voltage', 'Current');grid on;