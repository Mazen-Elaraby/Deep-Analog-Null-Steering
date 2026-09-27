% MATLAB Script to divide an angular domain (0-90 elev, 0-360 az)
% into N equal-solid-angle sectors, and then subdivide each
% sector into M equal-solid-angle subsectors.
%
% Now supports generalized band counts (e.g., for 16, 32 sectors).

clear all;
clc;
close all;

%% 1. Configuration
% ====================================================
% MAIN SECTOR CONFIGURATION
main_az_bands = 4;      
main_elev_bands = 2;  

% SUBSECTOR CONFIGURATION (Per Sector)
sub_az_bands = 4;
sub_elev_bands = 2; 
% ====================================================

filename_out = sprintf('sector_data_%d.mat', main_az_bands * main_elev_bands);

%% 2. Initialize Domain
main_elev_start = 0;
main_elev_end = 90;
main_az_start = 0;
main_az_end = 360;

sectors = [];
total_subsectors = 0;

%% 3. Calculate Main Sector Edges
fprintf('Starting Sector Calculation (%d x %d = %d Main Sectors)...\n', ...
    main_az_bands, main_elev_bands, main_az_bands*main_elev_bands);

% Azimuth: Linear division
az_step = (main_az_end - main_az_start) / main_az_bands;
main_az_edges = main_az_start:az_step:main_az_end;

% Elevation: Equal Solid Angle division
% Replaces findElevationSplit with generalized function
main_elev_edges = getEqualSolidAngleBoundaries(main_elev_start, main_elev_end, main_elev_bands);
fprintf('Main Elevation Edges: %s\n\n', mat2str(main_elev_edges, 4));

%% 4. Loop through Main Sectors and Subdivide
sector_id = 1;

for e = 1:main_elev_bands
    % Get main sector elevation edges
    elev_a = main_elev_edges(e);
    elev_b = main_elev_edges(e+1);
    
    for a = 1:main_az_bands
        % Get main sector azimuth edges
        az_a = main_az_edges(a);
        az_b = main_az_edges(a+1);
        
        % --- Calculate Main Sector Center ---
        center_elev = findScalarCentroid(elev_a, elev_b);
        center_az = (az_a + az_b) / 2;
        
        % Store Main Sector Info
        current_sector.SectorID = sector_id;
        current_sector.Edges.Elev = [elev_a, elev_b];
        current_sector.Edges.Az = [az_a, az_b];
        current_sector.Center.Elev = center_elev;
        current_sector.Center.Az = center_az;
        current_sector.Subsectors = []; 
        
        % --- Subdivide this Main Sector ---
        
        % Sub-Azimuth division
        sub_az_step = (az_b - az_a) / sub_az_bands;
        sub_az_edges = az_a:sub_az_step:az_b;
        
        % Sub-Elevation division (Equal Solid Angle)
        sub_elev_edges = getEqualSolidAngleBoundaries(elev_a, elev_b, sub_elev_bands);
        
        subsector_local_id = 1;
        for ee = 1:sub_elev_bands
            sub_elev_a = sub_elev_edges(ee);
            sub_elev_b = sub_elev_edges(ee+1);
            
            for aa = 1:sub_az_bands
                sub_az_a = sub_az_edges(aa);
                sub_az_b = sub_az_edges(aa+1);
                
                % Calculate Subsector Center
                sub_center_elev = findScalarCentroid(sub_elev_a, sub_elev_b);
                sub_center_az = (sub_az_a + sub_az_b) / 2;
                
                % Store Subsector Info
                total_subsectors = total_subsectors + 1;
                subsector.GlobalID = total_subsectors;
                subsector.LocalID = subsector_local_id;
                subsector.Edges.Elev = [sub_elev_a, sub_elev_b];
                subsector.Edges.Az = [sub_az_a, sub_az_b];
                subsector.Center.Elev = sub_center_elev;
                subsector.Center.Az = sub_center_az;
                
                % Note: Weights are not calculated here. 
                % You must run your weight calculation script on this struct later.
                subsector.Weights = []; 
                
                % Add to the parent sector's struct
                current_sector.Subsectors = [current_sector.Subsectors, subsector];
                
                subsector_local_id = subsector_local_id + 1;
            end
        end
        
        % Add this fully processed sector to the main list
        sectors = [sectors, current_sector];
        sector_id = sector_id + 1;
    end
end

fprintf('...Calculation Complete. Total sectors: %d. Total subsectors: %d.\n\n', length(sectors), total_subsectors);

%% 5. Verification
fprintf('=====================================================\n');
fprintf(' SOLID ANGLE VERIFICATION\n');
fprintf('=====================================================\n');
total_subs = length(sectors) * sub_az_bands * sub_elev_bands;
expected_sa = (2*pi) / total_subs; % Hemisphere / count
fprintf('Expected Solid Angle: %.8f steradians\n', expected_sa);

all_solid_angles = [];
for i = 1:length(sectors)
    s = sectors(i);
    for j = 1:length(s.Subsectors)
        sub = s.Subsectors(j);
        sa = calculateSolidAngle(sub.Edges.Elev(1), sub.Edges.Elev(2), sub.Edges.Az(1), sub.Edges.Az(2));
        all_solid_angles(end+1) = sa;
    end
end

fprintf('Mean Solid Angle:     %.8f sr\n', mean(all_solid_angles));
fprintf('StdDev Solid Angle:   %.2e\n', std(all_solid_angles));
if std(all_solid_angles) < 1e-6
    fprintf('>> VERIFICATION PASSED: Subsectors are equal-area.\n');
else
    fprintf('>> VERIFICATION FAILED: Check logic.\n');
end

%% 6. Save
save(filename_out, 'sectors');
fprintf('\nSaved to %s\n', filename_out);
fprintf('IMPORTANT: This file contains geometry only. Remember to calculate weights separately.\n');


