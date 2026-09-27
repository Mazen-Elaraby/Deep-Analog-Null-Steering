function centroid_deg = findScalarCentroid(theta_a_deg, theta_b_deg)
    % Calculates the scalar centroid of an elevation band [theta_a, theta_b]
    % using the formula:
    %   theta_bar = integral(theta * sin(theta)) / integral(sin(theta))
    
    % Convert to radians for calculation
    theta_a_rad = deg2rad(theta_a_deg);
    theta_b_rad = deg2rad(theta_b_deg);
    
    % Calculate the denominator: integral(sin(theta))
    % indefinite integral = -cos(theta)
    denominator = (-cos(theta_b_rad)) - (-cos(theta_a_rad));
    % This simplifies to: cos(theta_a_rad) - cos(theta_b_rad)
    
    % Calculate the numerator: integral(theta * sin(theta))
    % indefinite integral = sin(theta) - theta * cos(theta)
    val_b = sin(theta_b_rad) - theta_b_rad * cos(theta_b_rad);
    val_a = sin(theta_a_rad) - theta_a_rad * cos(theta_a_rad);
    numerator = val_b - val_a;
    
    % Avoid division by zero if the band is of zero width
    if denominator == 0
        centroid_rad = theta_a_rad;
    else
        centroid_rad = numerator / denominator;
    end
    
    % Convert final result back to degrees
    centroid_deg = rad2deg(centroid_rad);
end
