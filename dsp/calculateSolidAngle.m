function sa = calculateSolidAngle(elev_a_deg, elev_b_deg, az_a_deg, az_b_deg)
    % Calculates the solid angle (in steradians) for a given sector.
    % Formula: Omega = (phi_b - phi_a) * (cos(theta_a) - cos(theta_b))
    % All inputs must be in degrees and will be converted to radians.

    % Convert edges to radians
    elev_a_rad = deg2rad(elev_a_deg);
    elev_b_rad = deg2rad(elev_b_deg);
    az_a_rad = deg2rad(az_a_deg);
    az_b_rad = deg2rad(az_b_deg);

    % Calculate the two parts of the integral
    delta_phi = az_b_rad - az_a_rad;
    delta_cos_theta = cos(elev_a_rad) - cos(elev_b_rad);
    
    % Final solid angle
    sa = delta_phi * delta_cos_theta;
end