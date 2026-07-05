% function [thetahat, thetabnd, sohData] = iterSOH_EKF(vk, ik, deltat, xhatMinus, AhatSOC, sohData)
%
% Performs ONE iteration of the SOH parameter EKF (Dual EKF parameter filter).
%
% ── WHY xhatMinus AND AhatSOC ARE INPUTS ────────────────────────────────
%
%   Plett p.232 specifies that the parameter filter's output prediction
%   must use xhat^- (the state PRIOR, after time update but BEFORE the
%   measurement update of the SOC filter). Using xhat^+ (posterior) would
%   be one half-step wrong.
%
%   AhatSOC (= df/dx, the 3×3 state transition Jacobian) is already
%   computed inside iterEKF Step 0. Accepting it as an input avoids
%   recomputing it here — both filters must use the same Ahat.
%
%   Correct call order in runEKF_dual.m each timestep:
%     1. [zk, zkbnd, xhatMinus, AhatSOC, ekfData] = iterEKF(...)
%        → runs SOC time update (Step 1a) and full measurement update
%        → returns xhatMinus (state after Step 1a, before Step 2b)
%        → returns AhatSOC  (Ahat computed in Step 0)
%     2. [thetahat, thetabnd, sohData] = iterSOH_EKF(vk, ik, deltat, xhatMinus, AhatSOC, sohData)
%        → runs full parameter filter using xhatMinus for output prediction
%
% ── WHAT IS ESTIMATED ───────────────────────────────────────────────────
%   theta = [R0; Q]  — ONE joint 2×1 vector, ONE 2×2 covariance
%
% ── THE PHI MATRIX ──────────────────────────────────────────────────────
%   Phi_k = dx_k / dtheta  (3×2)
%
%   Answers: if theta = [R0; Q] changes slightly, how much does
%   x = [iR1; iR2; z] change?
%
%   Non-trivial because Q drives the SOC dynamics:
%     z_k = z_{k-1} - (dt/Q)*i_{k-1}
%   → changing Q changes z → changes OCV(z) → changes voltage.
%
%   Recursive update (Plett p.189):
%     Phi_k = df/dtheta + AhatSOC * Phi_{k-1}
%
%   df/dtheta  (3×2):
%     row 1 (iR1): [0,  0            ]   R0,Q absent from iR1 eq.
%     row 2 (iR2): [0,  0            ]   R0,Q absent from iR2 eq.
%     row 3 (z)  : [0,  dt*I_prev/Q²]   Q present: ∂(-dt/Q*i)/∂Q = +dt*i/Q²
%
% ── THE JACOBIAN Chat (total derivative, Plett p.188) ───────────────────
%
%   Chat = dg/dtheta = ∂g/∂theta  +  (∂g/∂x) * Phi_k    (1×2)
%
%   ∂g/∂theta (1×2):
%     ∂g/∂R0 = -ik   (R0 multiplies ik directly in g)
%     ∂g/∂Q  =  0    (Q absent from g explicitly)
%
%   ∂g/∂x (1×3):   same structure as Chat in SOC EKF
%     ∂g/∂iR1 = -R1
%     ∂g/∂iR2 = -R2
%     ∂g/∂z   = dOCV/dz
%
% ── SIX EKF STEPS (Plett Appendix p.229/232) ────────────────────────────
%   1a: theta^-  = theta^+_{k-1}         (random walk: mean unchanged)
%   1b: Sigma^-  = Sigma^+ + SigmaR      (covariance grows with drift)
%   1c: Phi_k updated; dhat computed using xhatMinus
%   2a: Chat, SigmaD, L
%   2b: theta^+ = theta^- + L*(vk - dhat)
%   2c: Sigma^+ = Sigma^- - L*SigmaD*L'
%
% Inputs:
%   vk         : measured terminal voltage [V]
%   ik         : measured current [A], positive = discharge
%   deltat     : sample interval [s]
%   xhatMinus  : [iR1^-; iR2^-; z^-] — state PRIOR from iterEKF Step 1a
%   AhatSOC    : 3×3 state transition Jacobian df/dx from iterEKF Step 0
%   sohData    : persistent data structure from initSOH_EKF
%
% Outputs:
%   thetahat : [R0_est; Q_est] updated parameter estimates
%   thetabnd : [3σ_R0; 3σ_Q]  3-sigma bounds
%   sohData  : updated data structure (pass to next call)

%% Last updated: June 30 2026

% function [thetahat, thetabnd, sohData] = iterSOH_EKF(vk, ik, deltat, xhatMinus, AhatSOC, sohData)
function [thetahat, thetabnd, sohData] = iterSOH_EKF(vk, ik, deltat, xhatMinus, AhatSOC, sohData, freeze)

  % ── Unpack ────────────────────────────────────────────────────────────
  model  = sohData.model;

  R0Ind  = sohData.R0Ind;
  QInd   = sohData.QInd;

  % Current parameter estimate (= theta^- after Step 1a, no change to mean)
  R0 = sohData.thetahat(R0Ind);
  Q  = sohData.thetahat(QInd);

  % R1 = model.R1;
  % R2 = model.R2;

  % State prior from SOC filter (after time update, before meas. update)
  iR1_m = xhatMinus(1);
  iR2_m = xhatMinus(2);
  z_m   = xhatMinus(3);

  R1 = interp1(model.SOC_R1, model.R1_tab, z_m, 'linear', 'extrap');
  R2 = interp1(model.SOC_all, model.R2_tab, z_m, 'linear', 'extrap');

  % i_{k-1}: needed for df/dtheta row 3
  I_prev = sohData.priorI;

  % ══════════════════════════════════════════════════════════════════════
  % STEP 1a — Parameter prediction time update
  %   theta^-_k = theta^+_{k-1}
  %   No code needed — sohData.thetahat already holds theta^-.
  % ══════════════════════════════════════════════════════════════════════

  % ══════════════════════════════════════════════════════════════════════
  % STEP 1b — Error covariance time update
  %   Sigma^- = Sigma^+ + SigmaR
  % ══════════════════════════════════════════════════════════════════════
  SigmaTheta = sohData.SigmaTheta + sohData.SigmaR;

  % ══════════════════════════════════════════════════════════════════════
  % STEP 1c — Update Phi and compute output prediction
  %
  %   df/dtheta (3×2): direct partial of each state eq. w.r.t. [R0, Q]
  %     Only z row / Q col is nonzero:
  %     ∂z_k/∂Q = +dt * i_{k-1} / Q²
  %     (from differentiating z_k = z_{k-1} - (dt/Q)*i_{k-1} w.r.t. Q)
  % ══════════════════════════════════════════════════════════════════════
  zkInd = 3;
  dfdtheta = zeros(3, 2);
  dfdtheta(zkInd, QInd) = deltat * I_prev / Q^2;

  % Recursive Phi update (Plett p.189):
  %   Phi_k = df/dtheta + AhatSOC * Phi_{k-1}
  %
  %   AhatSOC comes directly from iterEKF Step 0 — same matrix that
  %   propagates the SOC covariance. No recomputation needed here.
  Phi = dfdtheta + AhatSOC * sohData.Phi;   % 3×2

  % Output prediction using xhatMinus (state PRIOR, not posterior):
  %   dhat = OCV(z^-) - R1*iR1^- - R2*iR2^- - R0_est * ik
  OCV_z = interp1(model.SOC, model.OCV, z_m, 'linear', 'extrap');
  dhat  = OCV_z - R1*iR1_m - R2*iR2_m - R0*ik;

  if freeze
    sohData.Phi    = Phi;     % Phi accumulates during warmup
    sohData.priorI = ik;      % priorI updates every step always
    thetahat = sohData.thetahat;
    thetabnd = 3 * sqrt(diag(sohData.SigmaTheta));
    return;
  end

  % ══════════════════════════════════════════════════════════════════════
  % STEP 2a — Build Chat (total Jacobian, 1×2), then Kalman gain
  %
  %   Chat = ∂g/∂theta  +  (∂g/∂x) * Phi     (1×2)
  % ══════════════════════════════════════════════════════════════════════

  % ∂g/∂theta (1×2): direct partial
  dgdtheta = [-ik, 0];
  %   ∂g/∂R0 = -ik,   ∂g/∂Q = 0

  % ∂g/∂x (1×3): same as Chat in SOC EKF, evaluated at z^-
  dOCV_dz = dOCVfromSOC(model.SOC, model.OCV, z_m);
  dgdx    = [-R1, -R2, dOCV_dz];

  % Total derivative (1×2)
  Chat = dgdtheta + dgdx * Phi;

  % Dhat = dg/de = 1 (additive noise)
  Dhat   = 1;
  SigmaD = Chat * SigmaTheta * Chat' + Dhat^2 * sohData.SigmaE;   % scalar

  % Kalman gain (2×1) — corrects both R0 and Q jointly
  L = SigmaTheta * Chat' / SigmaD;

  % ══════════════════════════════════════════════════════════════════════
  % STEP 2b — Parameter estimate measurement update
  %   theta^+ = theta^- + L * (vk - dhat)
  % ══════════════════════════════════════════════════════════════════════
  r = vk - dhat;   % scalar voltage residual

  % Outlier rejection
  if r^2 > 100 * SigmaD
    L(:) = 0;
  end

  theta = sohData.thetahat + L * r;   % joint update of [R0; Q]

  % Physical bounds
  theta(R0Ind) = max(sohData.R0_nominal * 0.5, min(sohData.R0_nominal * 3.0, theta(R0Ind)));
  theta(QInd)  = max(sohData.Q_nominal  * 0.5, min(sohData.Q_nominal  * 1.035, theta(QInd)));
  % theta(R0Ind) = max(model.R0 * 0.5, min(model.R0 * 3.0, theta(R0Ind)));
  % theta(QInd)  = max(model.Q  * 0.1, min(model.Q  * 1.05, theta(QInd)));

  % ══════════════════════════════════════════════════════════════════════
  % STEP 2c — Error covariance measurement update
  %   Sigma^+ = Sigma^- - L * SigmaD * L'
  % ══════════════════════════════════════════════════════════════════════
  SigmaTheta = SigmaTheta - L * SigmaD * L';

  % Q-bump: if residual > 2σ, inflate Sigma(Q,Q) for faster Q adaptation
  if r^2 > 4 * SigmaD
    SigmaTheta(QInd, QInd) = SigmaTheta(QInd, QInd) * sohData.Qbump;
  end

  % Symmetrize and maintain positive semi-definiteness (same as iterEKF)
  [~, S, V]  = svd(SigmaTheta);
  HH         = V * S * V';
  SigmaTheta = (SigmaTheta + SigmaTheta' + HH + HH') / 4;

  % ── Save for next iteration ───────────────────────────────────────────
  sohData.thetahat   = theta;
  sohData.SigmaTheta = SigmaTheta;
  sohData.Phi        = Phi;      % Phi_k becomes Phi_{k-1} next step
  sohData.priorI     = ik;       % i_k  becomes i_{k-1}   next step

  % Keep model in sohData up to date (used in runEKF_dual to sync ekfData)
  sohData.model.R0 = theta(R0Ind);
  sohData.model.Q  = theta(QInd);

  % ── Outputs ──────────────────────────────────────────────────────────
  thetahat = theta;
  thetabnd = 3 * sqrt(diag(SigmaTheta));   % [3σ_R0; 3σ_Q]

end