% function sohData = initSOH_EKF(model, SigmaTheta0, SigmaR, SigmaE)
%
% Initialises the data structure used by iterSOH_EKF at every timestep.
%
% ── PARAMETER VECTOR ────────────────────────────────────────────────────
%   theta = [R0; Q]
%     R0 : series resistance [Ohm]  — rises with age (power fade)
%     Q  : total capacity   [A·s]   — falls with age (capacity fade)
%
%   ONE 2×1 vector. ONE 2×2 covariance. Never two separate filters.
%
% ── PHI MATRIX  (key concept) ───────────────────────────────────────────
%   Phi = dx_k / dtheta   (3×2)
%
%   Tracks how the cell states [iR1; iR2; z] would change if theta
%   changed. It is the reason the SOH Jacobian is a TOTAL derivative
%   rather than a simple partial derivative. Updated recursively every
%   step inside iterSOH_EKF:
%       Phi_k = df/dtheta + AhatSOC * Phi_{k-1}
%
%   Initialised to zeros per Plett p.189.
%
% Inputs:
%   model        : ESC model struct (fields: R0 R1 R2 C1 C2 Q SOC OCV)
%   SigmaTheta0  : 2×2 initial parameter estimation error covariance
%   SigmaR       : 2×2 process noise covariance (parameter drift rate)
%   SigmaE       : scalar measurement noise variance [V²]
%                  (use the same value as SigmaV in the SOC EKF)
%
% Output:
%   sohData : persistent data structure for iterSOH_EKF

%% Last updated: May 2026

function sohData = initSOH_EKF(model, SigmaTheta0, SigmaR, SigmaE)

  % ── Parameter indices ─────────────────────────────────────────────────
  sohData.R0Ind = 1;   % theta(1) = R0
  sohData.QInd  = 2;   % theta(2) = Q

  % ── Initial estimates = nominal model values ──────────────────────────
  sohData.thetahat = [model.R0;
                      model.Q];

  % ── Nominal values for SOH% reporting ────────────────────────────────
  sohData.R0_nominal = model.R0;
  sohData.Q_nominal  = model.Q;

  % ── Covariances ───────────────────────────────────────────────────────
  sohData.SigmaTheta = SigmaTheta0;   % 2×2 joint uncertainty on [R0; Q]
  sohData.SigmaR     = SigmaR;        % 2×2 process noise
  sohData.SigmaE     = SigmaE;        % scalar voltage noise variance [V²]

  % ── Qbump factor ──────────────────────────────────────────────────────
  % Inflate SigmaTheta(Q,Q) when voltage residual is persistently large.
  sohData.Qbump = 5;

  % ── Phi = dx_k/dtheta  (3×2) ─────────────────────────────────────────
  % Rows = states [iR1; iR2; z].  Cols = parameters [R0; Q].
  % Initialised to zeros (Plett p.189).
  % Carried forward between steps by iterSOH_EKF (Phi_k → Phi_{k-1}).
  sohData.Phi = zeros(3, 2);

  % ── Previous current i_{k-1} ─────────────────────────────────────────
  % Needed inside df/dtheta:  d(z_k)/dQ = dt * i_{k-1} / Q²
  sohData.priorI = 0;

  % ── Store model ───────────────────────────────────────────────────────
  sohData.model = model;

end
