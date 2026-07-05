% function dOCVz = dOCVfromSOC(SOC,OCV,z)
%     dZ = SOC(2) - SOC(1); % Find spacing of SOC vector
%     dUdZ = diff(OCV)/dZ; % Scaled forward finite difference
%     dOCV = ([dUdZ(1) dUdZ] + [dUdZ dUdZ(end)])/2; % Avg of fwd/bkwd diffs
%     dOCVz = interp1(SOC,dOCV,z);
% end

function dOCVz = dOCVfromSOC(SOC, OCV, z)
% dOCVfromSOC  Compute dOCV/dSOC at query point(s) z
%   Uses central differences for interior points,
%   and one-sided differences at the endpoints.

  SOC = SOC(:);   % force column vectors
  OCV = OCV(:);
  n   = length(SOC);

  dOCV = zeros(n, 1);

  % One-sided (forward) difference at first point
  dOCV(1) = (OCV(2) - OCV(1)) / (SOC(2) - SOC(1));

  % Central differences for interior points -- each uses its own spacing
  for i = 2 : n-1
    dOCV(i) = (OCV(i+1) - OCV(i-1)) / (SOC(i+1) - SOC(i-1));
  end

  % One-sided (backward) difference at last point
  dOCV(n) = (OCV(n) - OCV(n-1)) / (SOC(n) - SOC(n-1));

  % Interpolate to the query SOC value(s), extrapolate if needed
  dOCVz = interp1(SOC, dOCV, z, 'linear', 'extrap');
end