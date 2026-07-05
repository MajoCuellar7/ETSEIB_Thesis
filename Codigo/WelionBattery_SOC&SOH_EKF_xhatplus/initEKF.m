% function ekfData = initEKF(v0, SigmaX0, SigmaV, SigmaW, model)
%
% Initializes the SOC Extended Kalman Filter data structure.
%
% ── DUAL EKF COMPATIBILITY NOTE ─────────────────────────────────────────
%   In the dual EKF (runEKF_dual.m), the SOH filter estimates R0 and Q.
%   Before each call to iterEKF, runEKF_dual updates:
%       ekfData.model.R0  ← current R0 estimate from SOH filter
%       ekfData.model.Q   ← current Q  estimate from SOH filter
%   Because iterEKF reads R0 and Q from ekfData.model at the start of
%   every step, the coupling is automatic. This init function does NOT
%   need any structural change for dual EKF — the model stored here is
%   just the starting point, and it is overwritten at runtime.
%
% ── STATE VECTOR ────────────────────────────────────────────────────────
%   xhat = [iR1; iR2; z]
%     iR1 : RC branch 1 current [A]
%     iR2 : RC branch 2 current [A]
%     z   : state of charge (SOC) [-]
%
% ── MODEL ───────────────────────────────────────────────────────────────
%   No hysteresis. 2RC equivalent circuit.
%   State equations:
%     iR1_k = tau1*iR1_{k-1} + (1-tau1)*i_{k-1}
%     iR2_k = tau2*iR2_{k-1} + (1-tau2)*i_{k-1}
%     z_k   = z_{k-1} - (dt/Q)*i_{k-1}
%   Output equation:
%     v_k = OCV(z_k) - R1*iR1_k - R2*iR2_k - R0*i_k
%
% Inputs:
%   v0      : initial cell voltage [V] — used to estimate initial SOC
%   SigmaX0 : 3×3 initial state uncertainty covariance
%   SigmaV  : scalar measurement noise variance [V²]
%   SigmaW  : scalar process noise variance
%   model   : ESC cell model struct
%             Required fields: R0, R1, R2, C1, C2, Q, SOC, OCV
%
% Output:
%   ekfData : data structure consumed by iterEKF

%% Last updated: May 2026, adapted from Plett (2015)

function ekfData = initEKF(v0, SigmaX0, SigmaV, SigmaW, model)

  % ── State indices ─────────────────────────────────────────────────────
  % xhat = [iR1; iR2; z]
  ekfData.ir1Ind = 1;   % RC branch 1 current
  ekfData.ir2Ind = 2;   % RC branch 2 current
  ekfData.zkInd  = 3;   % SOC

  % ── Initial state ─────────────────────────────────────────────────────
  % Both RC branch currents start at zero (cell assumed at rest).
  % Initial SOC estimated by inverting the OCV curve at the measured v0.
  ir1_0 = 0;
  ir2_0 = 0;
  SOC0  = interp1(model.OCV, model.SOC, v0, 'linear', 'extrap');
  SOC0  = min(1, max(0, SOC0));

  ekfData.xhat = [ir1_0; ir2_0; SOC0];

  % ── Covariances ───────────────────────────────────────────────────────
  ekfData.SigmaX = SigmaX0;   % 3×3 state estimation error covariance
  ekfData.SigmaV = SigmaV;    % scalar: voltage sensor noise variance [V²]
  ekfData.SigmaW = SigmaW;    % scalar: process noise variance

  % Qbump: inflate SigmaX(z,z) when voltage residual is consistently large.
  % Keeps the SOC estimate from getting overconfident on bad measurements.
  ekfData.Qbump = 5;

  % ── Previous current ─────────────────────────────────────────────────
  % iterEKF uses i_{k-1} (priorI) for the state time update.
  % Initialized to 0 (cell assumed at rest before measurements begin).
  ekfData.priorI = 0;

  % ── Store model ───────────────────────────────────────────────────────
  % In the dual EKF, runEKF_dual.m updates ekfData.model.R0 and
  % ekfData.model.Q at every step with the latest SOH estimates before
  % calling iterEKF. So iterEKF always uses the most current parameter
  % values without any changes to iterEKF itself.
  ekfData.model = model;
end
