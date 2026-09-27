function edges = getEqualSolidAngleBoundaries(theta_start_deg, theta_end_deg, num_bands)
    % Generalizes findElevationSplit to N bands.
    % Divides the cosine domain into N equal segments.
    
    % 1. Convert to Cosine Domain (u = cos(theta))
    % Note: cos(0)=1, cos(90)=0. Function is decreasing.
    u_start = cosd(theta_start_deg);
    u_end = cosd(theta_end_deg);
    
    % 2. Create Linear Spacing in Cosine Domain
    u_edges = linspace(u_start, u_end, num_bands + 1);
    
    % 3. Convert back to Degrees
    edges = acosd(u_edges);
    
    % Ensure boundaries are exact (cleanup numerical noise)
    edges(1) = theta_start_deg;
    edges(end) = theta_end_deg;
end