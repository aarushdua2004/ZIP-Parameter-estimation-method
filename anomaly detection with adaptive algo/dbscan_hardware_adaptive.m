clc; clear; close all;

voltages = [402, 402, 402, 300, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 300, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 300, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 401, 401, 402, 401, 401, 402, 402, 402, 402, 401, 402, 401, 401, 402, 402, 402, 402, 401, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 401, 401, 401, 402, 402, 402, 402, 402, 402, 402, 402, 402, 500, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 401, 400, 385, 360, 352, 352, 250, 352, 342, 319, 305, 305, 305, 305, 305, 305, 305, 305, 310, 341, 355, 355, 355, 355, 355, 250, 397, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 402, 368, 351, 313, 303, 303, 303, 303, 303, 303, 303, 302, 302, 302, 302, 303, 313, 348, 353, 353, 353, 353, 354, 353, 250, 354, 355, 373, 401, 404, 411, 417, 416, 405, 402, 400, 399, 399, 399, 399, 399, 399, 399, 399, 399, 405, 405, 405, 405, 405, 404, 405, 405, 405, 405, 405, 405, 404, 404, 404, 405, 405, 405, 405, 405, 404, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 300, 405];
powers =[1160, 1160, 1160, 700, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 700, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1220, 1360, 1360, 1360, 1360, 1000, 1360, 1360, 1360, 1360, 1360, 1360, 1360, 1360, 1360, 1360, 1360, 1360, 1360, 1360, 1360, 1360, 1470, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1800, 1560, 1560, 1560, 1560, 1560, 1560, 1370, 1360, 1360, 1350, 1160, 1160, 1160, 800, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1150, 1120, 1090, 1090, 1090, 1090,700, 1070, 1060, 1060, 1060, 1060, 1060, 1060, 1060, 1060, 1060, 1060, 1080, 1090, 1090, 1090, 1090, 1090, 1100, 1140, 1160, 1170, 1170, 1170, 1170, 1170, 1170, 1170, 1170, 1170, 1170, 1170, 1170, 1170, 1170, 1170, 1170, 1170, 1170, 1170, 800, 1170, 1170, 1170, 1140, 1090, 1080, 1050, 1060, 1060, 1060, 1150, 1260, 1260, 1260, 1260, 1260, 1260, 1260, 1260, 1270, 1290, 1300, 1300, 1300, 1300, 1300, 1300, 1250, 1090, 1090, 1110, 1140, 1160, 1170, 1190, 1180, 1160, 1160, 1160, 1160, 1160, 1500, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1130, 959, 959, 959, 959, 959, 959, 959, 959, 959, 959, 959, 959, 959, 959, 959, 959, 959, 600, 959];

%% Base values - adjusted based on your new values
Vbase = 400;
Pbase = 1000; 

voltages_pu = voltages / Vbase;
powers_pu = powers / Pbase;
voltages_pu = voltages_pu(:);
powers_pu = powers_pu(:);
voltages = voltages(:);
powers = powers(:);

V0 = 1.0;

%% DBSCAN for Outlier Detection
features = [voltages_pu, powers_pu];

features_mean = mean(features);
features_std = std(features);
features_norm = (features - features_mean) ./ features_std;

% Set DBSCAN parameters
epsilon = 0.5; % Epsilon parameter for DBSCAN (neighborhood radius)
minPts = 5; % Minimum points to form a dense region

% Apply DBSCAN
try
    [idx, ~] = dbscan(features_norm, epsilon, minPts);
catch
    idx = manual_dbscan(features_norm, epsilon, minPts);
end

outliers = (idx == -1);
inliers = ~outliers;

fprintf('DBSCAN detected %d outliers out of %d total points (%.1f%%)\n', ...
    sum(outliers), length(outliers), sum(outliers)/length(outliers)*100);

%% Initial Least Squares ZIP Estimation - Using only inliers
valid_idx = inliers & (powers_pu > 0.01);
valid_v = voltages_pu(valid_idx);
valid_p = powers_pu(valid_idx);

H = [(valid_v / V0).^2, (valid_v / V0), ones(length(valid_v), 1)];

lambda = 0.001;
x = (H' * H + lambda * eye(3)) \ (H' * valid_p);

% Nominal power estimation
P0 = mean(valid_p);

Z_initial = x(1) / P0;
I_initial = x(2) / P0;
P_initial = x(3) / P0;

Z_initial = max(0, Z_initial);
I_initial = max(0, I_initial);
P_initial = max(0, P_initial);

sum_zip = Z_initial + I_initial + P_initial;
if sum_zip > 0
    Z_initial = Z_initial / sum_zip;
    I_initial = I_initial / sum_zip;
    P_initial = P_initial / sum_zip;
else
    
    Z_initial = 0.3;
    I_initial = 0.3;
    P_initial = 0.4;
end

fprintf('Initial ZIP values: Z=%.4f, I=%.4f, P=%.4f\n', Z_initial, I_initial, P_initial);

%% Try direct optimization for global parameters
options = optimset('Display', 'off', 'MaxIter', 1000);

function_handle = @(params) objective_function(params, voltages_pu, powers_pu, V0);

initial_guess = [Z_initial, I_initial, P0];

[optimized_params, ~] = fminsearch(function_handle, initial_guess, options);

Z_opt = optimized_params(1);
I_opt = optimized_params(2);
P_opt = 1 - Z_opt - I_opt;
P0_opt = optimized_params(3);

Z_opt = max(0, min(1, Z_opt));
I_opt = max(0, min(1, I_opt));
P_opt = max(0, min(1, P_opt));

sum_opt = Z_opt + I_opt + P_opt;
Z_opt = Z_opt / sum_opt;
I_opt = I_opt / sum_opt;
P_opt = P_opt / sum_opt;

fprintf('Optimized global ZIP values: Z=%.4f, I=%.4f, P=%.4f, P0=%.4f pu\n', Z_opt, I_opt, P_opt, P0_opt);

%% Dynamic Window Size Determination with DBSCAN insights
% Calculate the rate of change for voltage and power
dv_dt = diff(voltages_pu);
dp_dt = diff(powers_pu);

dv_dt = [0; dv_dt];
dp_dt = [0; dp_dt];

% Calculate moving standard deviation of changes
window_for_std = 3;

volatility_v = zeros(size(voltages_pu));
volatility_p = zeros(size(powers_pu));

for i = 1:length(voltages_pu)
    start_idx = max(1, i - floor(window_for_std/2));
    end_idx = min(length(voltages_pu), i + floor(window_for_std/2));
    window_data_v = voltages_pu(start_idx:end_idx);
    window_data_p = powers_pu(start_idx:end_idx);
    
    volatility_v(i) = std(window_data_v);
    volatility_p(i) = std(window_data_p);
end

max_vol_v = max(volatility_v);
max_vol_p = max(volatility_p);

if max_vol_v == 0
    norm_vol_v = zeros(size(volatility_v));
else
    norm_vol_v = volatility_v / max_vol_v;
end

if max_vol_p == 0
    norm_vol_p = zeros(size(volatility_p));
else
    norm_vol_p = volatility_p / max_vol_p;
end

% Combined volatility metric
combined_volatility = norm_vol_v + norm_vol_p;

% Add outlier penalty to volatility (higher window size for outliers to smooth them)
outlier_penalty = zeros(size(combined_volatility));
outlier_penalty(outliers) = 0.5;

min_window_size = 3;
max_window_size = 10;

dynamic_window_sizes = round(max_window_size - (max_window_size - min_window_size) * (combined_volatility - outlier_penalty));

dynamic_window_sizes = max(min_window_size, min(max_window_size, dynamic_window_sizes));

fprintf('Dynamic window sizes: Min=%d, Max=%d, Mean=%.2f\n', ...
    min(dynamic_window_sizes), max(dynamic_window_sizes), mean(dynamic_window_sizes));

%% Adaptive ZIP Estimation Using Dynamic Window Sizing and DBSCAN insights
P_calc_all = zeros(length(powers), 1);
zip_full = zeros(length(powers), 4); % [Z, I, P, P0]

for i = 1:length(powers)
    zip_full(i, :) = [Z_opt, I_opt, P_opt, P0_opt];
end

for i = 1:length(powers)
    window_size = dynamic_window_sizes(i);
    
    half_window = floor(window_size / 2);
    start_idx = max(1, i - half_window);
    end_idx = min(length(powers), i + half_window);
    
    if end_idx - start_idx + 1 < window_size
        start_idx = max(1, end_idx - window_size + 1);
    end
    
    V_window = voltages_pu(start_idx:end_idx);
    P_window = powers_pu(start_idx:end_idx);
    window_outliers = outliers(start_idx:end_idx);
    
    weights = ones(size(window_outliers));
    weights(window_outliers) = 0.3;  % Lower weight for outliers
    
    if any(isnan(V_window)) || any(isnan(P_window)) || all(P_window < 0.01)
        continue;
    end
    
    if outliers(i) && sum(~window_outliers) < 3
        continue;
    end
    
    try
        window_obj_func = @(params) weighted_objective_function(params, V_window, P_window, V0, weights);
        
        [opt_result, ~] = fminsearch(window_obj_func, [Z_opt, I_opt, P0_opt], options);
        
        z_win = max(0, min(1, opt_result(1)));
        i_win = max(0, min(1, opt_result(2)));
        p_win = 1 - z_win - i_win;
        p0_win = opt_result(3);
        
        p0_win = max(0.1, p0_win);
        
        sum_win = z_win + i_win + p_win;
        z_win = z_win / sum_win;
        i_win = i_win / sum_win;
        p_win = p_win / sum_win;
        
        zip_full(i, :) = [z_win, i_win, p_win, p0_win];
    catch
    end
end

%% Apply local smoothing to reduce parameter oscillations
smoothing_window = 3;
for i = 1:length(powers)
    start_smooth = max(1, i - floor(smoothing_window/2));
    end_smooth = min(length(powers), i + floor(smoothing_window/2));
    
    if outliers(i)
        smoothing_window_outlier = 5;
        start_smooth = max(1, i - floor(smoothing_window_outlier/2));
        end_smooth = min(length(powers), i + floor(smoothing_window_outlier/2));
    end
    
    zip_full(i, :) = mean(zip_full(start_smooth:end_smooth, :), 1);
end

%% Calculate power with the ZIP model
for i = 1:length(powers)
    Z_val = zip_full(i, 1);
    I_val = zip_full(i, 2);
    P_val = zip_full(i, 3);
    P0_val = zip_full(i, 4);
    
    v_ratio = voltages_pu(i) / V0;
    P_calc_all(i) = P0_val * (Z_val * v_ratio^2 + I_val * v_ratio + P_val) * Pbase;
end

%% Calculate individual components for visualization
Z_component = zip_full(:,1) .* (voltages_pu/V0).^2 .* zip_full(:,4) * Pbase;
I_component = zip_full(:,2) .* (voltages_pu/V0) .* zip_full(:,4) * Pbase;
P_component = zip_full(:,3) .* zip_full(:,4) * Pbase;

%% Create calculation expressions
P_calc_expression = strings(length(powers), 1);
Z_calc_expression = strings(length(powers), 1);
I_calc_expression = strings(length(powers), 1);
P_param_expression = strings(length(powers), 1);

for i = 1:length(powers)
    P_calc_expression(i) = sprintf('(%.4f*(%.3f/%.3f)^2 + %.4f*(%.3f/%.3f) + %.4f)%.2f%.0f', ...
        zip_full(i,1), voltages_pu(i), V0, ...
        zip_full(i,2), voltages_pu(i), V0, ...
        zip_full(i,3), zip_full(i,4), Pbase);
        Z_calc_expression(i) = sprintf('%.4f*(%.3f/%.3f)^2*%.2f*%.0f = %.2f W', ...
        zip_full(i,1), voltages_pu(i), V0, zip_full(i,4), Pbase, Z_component(i));
    
    I_calc_expression(i) = sprintf('%.4f*(%.3f/%.3f)%.2f%.0f = %.2f W', ...
        zip_full(i,2), voltages_pu(i), V0, zip_full(i,4), Pbase, I_component(i));
    
    P_param_expression(i) = sprintf('%.4f*%.2f*%.0f = %.2f W', ...
        zip_full(i,3), zip_full(i,4), Pbase, P_component(i));
end

%% Final Table Output
num_rows = length(powers);
indices = (1:num_rows)';
window_sizes = dynamic_window_sizes(1:num_rows);

outlier_status = cell(num_rows, 1);
for i = 1:num_rows
    if outliers(i)
        outlier_status{i} = 'Outlier';
    else
        outlier_status{i} = 'Normal';
    end
end

T = table(indices, voltages, powers, P_calc_all, P_calc_expression, ...
    zip_full(:,1), Z_calc_expression, ...
    zip_full(:,2), I_calc_expression, ...
    zip_full(:,3), P_param_expression, ...
    zip_full(:,4) * Pbase, ...
    window_sizes, ...
    outlier_status, ...
    'VariableNames', {'Index', 'Voltage', 'Power_measured', 'P_calc', 'P_calc_Formula', ...
                     'Z_coeff', 'Z_Component', ...
                     'I_coeff', 'I_Component', ...
                     'P_coeff', 'P_Component', ...
                     'P0_scaled', ...
                     'Window_Size', ...
                     'Status'});

% Display first few rows - use head function if available, otherwise just display first 5 rows
disp(T);

%% Plots
figure;
subplot(3,1,1);
plot(powers, 'b', 'LineWidth', 1.5, 'DisplayName', 'Measured Power');
hold on;
plot(P_calc_all, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Calculated Power (ZIP)');
% Highlight outliers
if sum(outliers) > 0
    scatter(find(outliers), powers(outliers), 50, 'ko', 'filled', 'DisplayName', 'Outliers');
end
xlabel('Sample Index');
ylabel('Power (W)');
legend('Location', 'best');
title('Measured vs Estimated Power');
grid on;

subplot(3,1,2);
plot(voltages, 'k-', 'LineWidth', 1.5);
hold on;
% Highlight outliers
if sum(outliers) > 0
    scatter(find(outliers), voltages(outliers), 50, 'ro', 'filled', 'DisplayName', 'Outliers');
    legend('Voltage', 'Outliers', 'Location', 'best');
end
xlabel('Sample Index');
ylabel('Voltage (V)');
title('Voltage Measurements');
grid on;

subplot(3,1,3);
plot(window_sizes, 'g-', 'LineWidth', 1.5);
xlabel('Sample Index');
ylabel('Window Size');
title('Dynamic Window Sizes');
grid on;

% Plot DBSCAN clusters
figure;
scatter(features_norm(:,1), features_norm(:,2), 50, idx, 'filled');
colormap(jet);
colorbar;
hold on;
% Highlight outliers
if sum(outliers) > 0
    scatter(features_norm(outliers,1), features_norm(outliers,2), 80, 'ko', 'LineWidth', 2);
end
title('DBSCAN Clustering Results');
xlabel('Normalized Voltage');
ylabel('Normalized Power');
grid on;
legend('Clusters', 'Outliers', 'Location', 'best');

%% Define the objective function here (nested function)
function error = objective_function(params, voltages, powers, V0)
        Z = params(1);
        I = params(2);
        P0 = params(3);     
        P = 1 - Z - I;
        
        if Z < 0 || I < 0 || P < 0 || P0 <= 0
            error = 1e10; 
            return;
        end
        
        P_est = P0 * (Z * (voltages / V0).^2 + I * (voltages / V0) + P);
        
        error = mean((P_est - powers).^2);
end

%% Define weighted objective function
function error = weighted_objective_function(params, voltages, powers, V0, weights)
    Z = params(1);
    I = params(2);
    P0 = params(3);
    P = 1 - Z - I;
    
    if Z < 0 || I < 0 || P < 0 || P0 <= 0
        error = 1e10;
        return;
    end
    
    P_est = P0 * (Z * (voltages / V0).^2 + I * (voltages / V0) + P);
    
    squared_errors = (P_est - powers).^2;
    error = sum(weights .* squared_errors) / sum(weights);
end

%% Manual implementation of DBSCAN
function idx = manual_dbscan(X, epsilon, minPts)
    n = size(X, 1);
    idx = zeros(n, 1) - 1; 
    cluster_id = 0;
    dist_matrix = pdist2(X, X);
    
    for i = 1:n
        if idx(i) ~= -1
            continue;
        end
        neighbors = find(dist_matrix(i, :) <= epsilon);
        
        if length(neighbors) < minPts
            idx(i) = -1;
        else
            cluster_id = cluster_id + 1;
            idx = expand_cluster(X, i, neighbors, cluster_id, idx, dist_matrix, epsilon, minPts);
        end
    end
end

%% Helper function for manual DBSCAN implementation
function idx = expand_cluster(X, point_idx, neighbors, cluster_id, idx, dist_matrix, epsilon, minPts)
    idx(point_idx) = cluster_id;
    
    i = 1;
    while i <= length(neighbors)
        current_point = neighbors(i);
        
        if idx(current_point) == -1
            idx(current_point) = cluster_id;
        end
        
        if idx(current_point) == 0
            idx(current_point) = cluster_id;
            
            current_neighbors = find(dist_matrix(current_point, :) <= epsilon);
            
            if length(current_neighbors) >= minPts
                neighbors = [neighbors, current_neighbors];
            end
        end
        
        i = i + 1;
    end
    
    idx = idx;
end