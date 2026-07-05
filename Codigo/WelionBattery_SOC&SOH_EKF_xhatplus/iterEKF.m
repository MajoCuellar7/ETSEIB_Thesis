% function [zk,zkbnd,ekfData] = iterEKF(vk,ik,Tk,deltat,ekfData)
%
%    Performs one iteration of the extended Kalman filter using the new
%    measured data.
%
% Inputs:
%   vk: Present measured (noisy) cell voltage
%   ik: Present measured (noisy) cell current
%   Tk: Present temperature
%   deltat: Sampling interval
%   ekfData: Data structure initialized by initEKF, updated by iterEKF
%
% Output:
%   zk: SOC estimate for this time sample
%   zkbnd: 3-sigma estimation bounds
%   ekfData: Data structure used to store persistent variables
 
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

%% Last updated: July 2nd 2026, Maria J. Cuellar

function [zk, zkbnd, xhatMinus, AhatSOC, ekfData] = iterEKF(vk, ik, deltat, ekfData)

    % REEMPLAZA las 5 líneas de "model.R0", "model.R1", etc. con esto:
    model = ekfData.model;

    R0 = model.R0;

    z_now = min(1, max(0, ekfData.xhat(ekfData.zkInd)));  % SOC del paso anterior
    
    % R0 = interp1(model.SOC_all, model.R0_tab, z_now, 'linear', 'extrap');
    R1 = interp1(model.SOC_R1,  model.R1_tab, z_now, 'linear', 'extrap');
    R2 = interp1(model.SOC_all, model.R2_tab, z_now, 'linear', 'extrap');
    C1 = interp1(model.SOC_R1,  model.C1_tab, z_now, 'linear', 'extrap');
    C2 = interp1(model.SOC_all, model.C2_tab, z_now, 'linear', 'extrap');
    Q  = model.Q;
  
  % --- Discrete-time RC decay factors ---
  tau1 = exp(-deltat / (R1 * C1));
  tau2 = exp(-deltat / (R2 * C2));

  % Get data stored in ekfData structure
  I = ekfData.priorI; % current from PREVIOUS time step
  SigmaX = ekfData.SigmaX; % estimation covariance
  SigmaV = ekfData.SigmaV;
  SigmaW = ekfData.SigmaW;
  xhat = ekfData.xhat;

  ir1Ind = ekfData.ir1Ind;
  ir2Ind = ekfData.ir2Ind;
  zkInd = ekfData.zkInd;
  
  % EKF Step 0: Compute Ahat[k-1], Bhat[k-1]
  nx = length(xhat);
  Ahat = zeros(nx,nx);
  Ahat(ir1Ind, ir1Ind) = tau1;
  Ahat(ir2Ind, ir2Ind) = tau2;
  Ahat(zkInd, zkInd) = 1;
  AhatSOC = Ahat; % return to SOH filter

  % Bhat = df/dw = [1-tau1; 1-tau2; -deltat/Q]
  Bhat = zeros(nx,1);
  Bhat(ir1Ind) = 1 - tau1;
  Bhat(ir2Ind) = 1 - tau2;
  Bhat(zkInd) = -deltat / Q;
  
  % Step 1a: State estimate time update
  % uses Prior current I
  % order or second order system, so if it's first order R2 is only 0.
  xhat(ir1Ind) = tau1 * xhat(ir1Ind) + (1 - tau1)* I;
  xhat(ir2Ind) = tau2 * xhat(ir2Ind) + (1 - tau2)* I;
  xhat(zkInd) = xhat(zkInd) - (deltat / Q) * I;

  xhat(zkInd) = min(1.05, max(-0.05, xhat(zkInd))); % limits to [1.05, -0.05]
  %The small buffer beyond [0,1] is intentional — it allows the filter to temporarily go slightly outside the physical range without hard-clipping, which can cause numerical problems in the covariance update. The filter will naturally pull back within [0,1] through the measurement updates. Plett uses this same buffer. It does NOT mean the battery is more charged than full — it is just a numerical safety margin.
  xhatMinus = xhat; % state PRIOR
  
  % Step 1b: Error covariance time update
  SigmaX = Ahat * SigmaX * Ahat' + Bhat * SigmaW * Bhat';
  
  % Step 1c: Output estimate
  OCV_z = interp1(model.SOC, model.OCV, xhat(zkInd), 'linear', 'extrap');
  yhat = OCV_z - R1 * xhat(ir1Ind) - R2 * xhat(ir2Ind) - R0 * ik;

  % Step 2a: Estimator gain matrix
  Chat = zeros(1, nx);
  Chat(zkInd) = dOCVfromSOC(model.SOC, model.OCV, xhat(zkInd));
  Chat(ir1Ind) = -R1;
  Chat(ir2Ind) = -R2;
  
  Dhat = 1;
  SigmaY = Chat * SigmaX * Chat' + Dhat * SigmaV * Dhat';
  L = SigmaX * Chat'/ SigmaY;
  
  % Step 2b: State estimate measurement update
  r = vk - yhat; % residual erros?
  if r^2 > 100 * SigmaY
    L(:)= 0.0;
  end 

  xhat = xhat + L * r;
  xhat(zkInd) = min(1.05, max(-0.05, xhat(zkInd)));
  
  % Step 2c: Error covariance measurement update
  SigmaX = SigmaX - L * SigmaY * L';

  % Q-bump code (this is in case the residual error is big
  if r^2 > 4 * SigmaY % bad voltage estimate by 2 std. devs, bump Q 
    fprintf('Bumping SigmaX\n');
    SigmaX(zkInd, zkInd) = SigmaX(zkInd, zkInd) * ekfData.Qbump;
  end

  [~, S, V] = svd(SigmaX);
  HH = V * S * V';
  SigmaX = (SigmaX + SigmaX' + HH + HH')/4; % Help maintain robustness
  
  % Save data in ekfData structure for next time...
  ekfData.priorI = ik;
  ekfData.SigmaX = SigmaX;
  ekfData.xhat = xhat;

  % --- Outputs ---
  zk = xhat(zkInd);
  zkbnd = 3 * sqrt(SigmaX(zkInd, zkInd));
end