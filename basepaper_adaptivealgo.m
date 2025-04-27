% Given Data
voltages = [1.0355; 1.0116; 0.9894; 1.0613; 1.0613; 1.0613; 1.0613; 1.0613; 1.0355; 1.0355; 
            1.0355; 1.0354; 1.0354; 1.0610; 1.0610; 1.0351; 1.0351; 1.0351; 1.0349; 1.0349; 1.0349];
powers = [0.1490; 0.1490; 0.1490; 0.1490; 0.1490; 0.1490; 0.1490; 0.1490; 0.1490; 0.1490; 
          0.1490; 0.1501; 0.1501; 0.1508; 0.1508; 0.1517; 0.1517; 0.1517; 0.1527; 0.1527; 0.1527];

V0 = 1;
% Use least squares to initialize ZIP values
% Create the H matrix for least squares
H = [(voltages/V0).^2, (voltages/V0), ones(length(voltages), 1)];
Z = powers;

% Solve for x = [Z*P0, I*P0, P*P0] using least squares method
x = (H'*H)\(H'*Z); P0 = 0.149;

% Define search grid for ZIP values
zip_range = 0:0.01:1; 
window_size = 3;
num_windows = length(voltages) - window_size + 1;

% Calculate the initial ZIP values
Z_initial = x(1)/P0;
I_initial = x(2)/P0;
P_initial = x(3)/P0;

sum_zip = Z_initial + I_initial + P_initial;
Z_initial = Z_initial / sum_zip;
I_initial = I_initial / sum_zip;
P_initial = P_initial / sum_zip;

% Display initialized values
fprintf('Initialized V0: %.4f\n', V0);
fprintf('Initialized P0: %.4f\n', P0);
fprintf('Initialized ZIP values: Z=%.4f, I=%.4f, P=%.4f\n', Z_initial, I_initial, P_initial);

% Adaptive search
for i = 1:num_windows
    V_window = voltages(i:i+window_size-1);
    P_window = powers(i:i+window_size-1);
    
    ZIP = [0.33, 0.33, 0.33]; 
    
    min_error = inf;
    best_zip = ZIP;
    
    for Z = zip_range
        for I = zip_range
            P = 1 - Z - I;
            if P < 0 || P > 1
                continue;
            end
            
            % Compute estimated power for the window
            P_calc = (Z * (V_window/V0).^2 + I * (V_window/V0) + P) * P0;
            
            error = sum((P_calc - P_window).^2);
            
            if error < min_error
                min_error = error;
                best_zip = [Z, I, P];
            end
        end
    end
    
    best_zip_values(i, :) = best_zip;
end

disp('Optimized ZIP Parameters for each window:');
disp(best_zip_values);

%% Perform Linear Regression to Find Global ZIP Model
X = [ones(num_windows,1), best_zip_values(:,1:2)]; 
y = best_zip_values(:,3); 

% Solve linear regression (Z and I as independent variables, P as dependent)
zip_global = X \ y;

Z_global = zip_global(2);
I_global = zip_global(3);
P_global = 1 - (Z_global + I_global);

fprintf('Global ZIP Parameters (Linear Regression):\n');
fprintf('Z = %.4f, I = %.4f, P = %.4f\n', Z_global, I_global, P_global);