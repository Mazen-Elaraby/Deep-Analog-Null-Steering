function theta_split = findElevationSplit(theta_a_deg, theta_b_deg)
    % Finds the elevation angle 'theta_split' that divides the solid
    % angle of the elevation band [theta_a, theta_b] into two equal halves.
    % This is found by finding the midpoint of the cosine values.

    % Convert to radians for calculation
    theta_a_rad = deg2rad(theta_a_deg);
    theta_b_rad = deg2rad(theta_b_deg);

    % Find the mean of the cosine values
    cos_mid = (cos(theta_a_rad) + cos(theta_b_rad)) / 2;

    % Convert back to an angle
    theta_split_rad = acos(cos_mid);

    % Convert back to degrees
    theta_split = rad2deg(theta_split_rad);
end

% function theta_splits = findElevationSplits(theta_a_deg, theta_b_deg, N)
%     % Finds the elevation angles 'theta_splits' that divide the solid
%     % angle of the elevation band [theta_a, theta_b] into N equal parts.
%     % This generalizes the bisection by finding equally spaced points in the cosine domain.
%     % theta_splits will be a row vector of N-1 split angles between theta_a and theta_b.
%     % Convert to radians for calculation
%     theta_a_rad = deg2rad(theta_a_deg);
%     theta_b_rad = deg2rad(theta_b_deg);
%     % Compute the cosine values
%     cos_a = cos(theta_a_rad);
%     cos_b = cos(theta_b_rad);
%     % The total delta in cosine
%     delta_cos_total = cos_a - cos_b;  % Assuming theta_a < theta_b, so cos_a > cos_b
%     % The step in cosine for each equal solid angle part
%     delta_cos = delta_cos_total / N;
%     % Compute the cosine values for the splits
%     cos_splits = cos_a - (1:(N-1)) * delta_cos;
%     % Convert back to angles in radians
%     theta_splits_rad = acos(cos_splits);
%     % Convert back to degrees
%     theta_splits = rad2deg(theta_splits_rad);
% end