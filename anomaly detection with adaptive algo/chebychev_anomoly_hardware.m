clc; clear; close all;

voltages = [402, 402, 402, 300, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 300, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 300, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 401, 401, 402, 401, 401, 402, 402, 402, 402, 401, 402, 401, 401, 402, 402, 402, 402, 401, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 401, 401, 401, 402, 402, 402, 402, 402, 402, 402, 402, 402, 500, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 401, 400, 385, 360, 352, 352, 250, 352, 342, 319, 305, 305, 305, 305, 305, 305, 305, 305, 310, 341, 355, 355, 355, 355, 355, 250, 397, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 402, 368, 351, 313, 303, 303, 303, 303, 303, 303, 303, 302, 302, 302, 302, 303, 313, 348, 353, 353, 353, 353, 354, 353, 250, 354, 355, 373, 401, 404, 411, 417, 416, 405, 402, 400, 399, 399, 399, 399, 399, 399, 399, 399, 399, 405, 405, 405, 405, 405, 404, 405, 405, 405, 405, 405, 405, 404, 404, 404, 405, 405, 405, 405, 405, 404, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 300, 405];
powers =[1160, 1160, 1160, 700, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 700, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1220, 1360, 1360, 1360, 1360, 1000, 1360, 1360, 1360, 1360, 1360, 1360, 1360, 1360, 1360, 1360, 1360, 1360, 1360, 1360, 1360, 1360, 1470, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1800, 1560, 1560, 1560, 1560, 1560, 1560, 1370, 1360, 1360, 1350, 1160, 1160, 1160, 800, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1150, 1120, 1090, 1090, 1090, 1090,700, 1070, 1060, 1060, 1060, 1060, 1060, 1060, 1060, 1060, 1060, 1060, 1080, 1090, 1090, 1090, 1090, 1090, 1100, 1140, 1160, 1170, 1170, 1170, 1170, 1170, 1170, 1170, 1170, 1170, 1170, 1170, 1170, 1170, 1170, 1170, 1170, 1170, 1170, 1170, 800, 1170, 1170, 1170, 1140, 1090, 1080, 1050, 1060, 1060, 1060, 1150, 1260, 1260, 1260, 1260, 1260, 1260, 1260, 1260, 1270, 1290, 1300, 1300, 1300, 1300, 1300, 1300, 1250, 1090, 1090, 1110, 1140, 1160, 1170, 1190, 1180, 1160, 1160, 1160, 1160, 1160, 1500, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1130, 959, 959, 959, 959, 959, 959, 959, 959, 959, 959, 959, 959, 959, 959, 959, 959, 959, 600, 959];

%% Local Anomaly Detection

% Initialize anomaly flags
anomaly_flags_voltage = false(size(voltages));
anomaly_flags_power = false(size(powers));

% Parameters for local anomaly detection
local_window_size = 11;
sensitivity_voltage = 2.5;
sensitivity_power = 1.5;

% Perform local anomaly detection using sliding window
for i = 1:length(voltages)
    % Define local window boundaries
    start_idx = max(1, i - floor(local_window_size/2));
    end_idx = min(length(voltages), i + floor(local_window_size/2));
    
    % Extract local windows
    local_voltages = voltages(start_idx:end_idx);
    local_powers = powers(start_idx:end_idx);
    
    % Skip current point for calculating local statistics
    mask_except_current = true(size(local_voltages));
    mask_except_current(i - start_idx + 1) = false;
    
    % Calculate local statistics for voltage and power
    local_v_mean = mean(local_voltages(mask_except_current));
    local_v_std = std(local_voltages(mask_except_current));
    
    local_p_mean = mean(local_powers(mask_except_current));
    local_p_std = std(local_powers(mask_except_current));
    
    % Check if current point is a local anomaly
    if abs(voltages(i) - local_v_mean) > sensitivity_voltage * local_v_std
        anomaly_flags_voltage(i) = true;
    end
    
    if abs(powers(i) - local_p_mean) > sensitivity_power * local_p_std
        anomaly_flags_power(i) = true;
    end
end

% Apply second-level filtering to reduce false positives
for i = 2:length(voltages)-1
    % check if it's an isolated anomaly
    if anomaly_flags_voltage(i) && ~anomaly_flags_voltage(i-1) && ~anomaly_flags_voltage(i+1)
        start_idx = max(1, i - floor(local_window_size/2));
        end_idx = min(length(voltages), i + floor(local_window_size/2));
        local_voltages = voltages(start_idx:end_idx);
        mask_except_current = true(size(local_voltages));
        mask_except_current(i - start_idx + 1) = false;
        local_v_mean = mean(local_voltages(mask_except_current));
        local_v_std = std(local_voltages(mask_except_current));
        
        if abs(voltages(i) - local_v_mean) <= sensitivity_voltage * 1.5 * local_v_std
            anomaly_flags_voltage(i) = false;
        end
    end
    
    if anomaly_flags_power(i) && ~anomaly_flags_power(i-1) && ~anomaly_flags_power(i+1)
        start_idx = max(1, i - floor(local_window_size/2));
        end_idx = min(length(powers), i + floor(local_window_size/2));
        local_powers = powers(start_idx:end_idx);
        mask_except_current = true(size(local_powers));
        mask_except_current(i - start_idx + 1) = false;
        local_p_mean = mean(local_powers(mask_except_current));
        local_p_std = std(local_powers(mask_except_current));
        
        if abs(powers(i) - local_p_mean) <= sensitivity_power * 1.5 * local_p_std
            anomaly_flags_power(i) = false;
        end
    end
end

% check for global anomalies (to catch extreme values)
global_v_mean = mean(voltages);
global_v_std = std(voltages);
global_p_mean = mean(powers);
global_p_std = std(powers);

global_thresh_v = 3.5; 
global_thresh_p = 3.0;

for i = 1:length(voltages)
    if abs(voltages(i) - global_v_mean) > global_thresh_v * global_v_std
        anomaly_flags_voltage(i) = true;
    end
    
    if abs(powers(i) - global_p_mean) > global_thresh_p * global_p_std
        anomaly_flags_power(i) = true;
    end
end

%% converting to pu values
Vbase = 400;
Pbase = 1000;

voltages_pu = voltages / Vbase;
powers_pu = powers / Pbase;
voltages_pu = voltages_pu(:);
powers_pu = powers_pu(:);
voltages = voltages(:);
powers = powers(:);

V0 = 1.0;

%% Initial Least Squares ZIP Estimation - Fixed to avoid NaNs
% Skip data points with zero or very small values and detected anomalies

anomaly_flags_voltage = anomaly_flags_voltage(:);
anomaly_flags_power = anomaly_flags_power(:);

valid_idx = powers_pu > 0.01 & ~anomaly_flags_power;
valid_v = voltages_pu(valid_idx);
valid_p = powers_pu(valid_idx);

H = [(valid_v / V0).^2, (valid_v / V0), ones(length(valid_v), 1)];

lambda = 0.001;
x = (H' * H + lambda * eye(3)) \ (H' * valid_p);

% Nominal power estimation
P0 = median(valid_p);

Z_initial = x(1) / P0;
I_initial = x(2) / P0;
P_initial = x(3) / P0;

Z_initial = max(0, Z_initial);
I_initial = max(0, I_initial);
P_initial = max(0, P_initial);

% Normalize to sum to 1
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

%% Dynamic Window Size Determination Based on Local Characteristics

dv_dt = diff(voltages_pu);
dp_dt = diff(powers_pu);

dv_dt = [0; dv_dt];
dp_dt = [0; dp_dt];

window_for_std = 5; 

volatility_v = zeros(size(voltages_pu));
volatility_p = zeros(size(powers_pu));

for i = 1:length(voltages_pu)
    start_idx = max(1, i - floor(window_for_std/2));
    end_idx = min(length(voltages_pu), i + floor(window_for_std/2));
    window_data_v = voltages_pu(start_idx:end_idx);
    window_data_p = powers_pu(start_idx:end_idx);
    
    window_anomalies_v = anomaly_flags_voltage(start_idx:end_idx);
    window_anomalies_p = anomaly_flags_power(start_idx:end_idx);
    
    clean_window_v = window_data_v(~window_anomalies_v);
    clean_window_p = window_data_p(~window_anomalies_p);
    
    if length(clean_window_v) > 1
        volatility_v(i) = std(clean_window_v);
    else
        volatility_v(i) = 0;
    end
    
    if length(clean_window_p) > 1
        volatility_p(i) = std(clean_window_p);
    else
        volatility_p(i) = 0;
    end
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

combined_volatility = norm_vol_v + norm_vol_p;

min_window_size = 3;
max_window_size = 10;

dynamic_window_sizes = round(max_window_size - (max_window_size - min_window_size) * combined_volatility);

anomaly_buffer = 4;
for i = 1:length(powers)
    if anomaly_flags_power(i) || anomaly_flags_voltage(i)
        dynamic_window_sizes(i) = max_window_size;
        
        for j = max(1, i-anomaly_buffer):min(length(powers), i+anomaly_buffer)
            dynamic_window_sizes(j) = max(dynamic_window_sizes(j), ceil(max_window_size * 0.8));
        end
    end
end

dynamic_window_sizes = max(min_window_size, min(max_window_size, dynamic_window_sizes));

fprintf('Dynamic window sizes: Min=%d, Max=%d, Mean=%.2f\n', ...
    min(dynamic_window_sizes), max(dynamic_window_sizes), mean(dynamic_window_sizes));

%% Adaptive ZIP Estimation Using Dynamic Window Sizing
P_calc_all = zeros(length(powers), 1);
zip_full = zeros(length(powers), 4); % [Z, I, P, P0]

for i = 1:length(powers)
    zip_full(i, :) = [Z_opt, I_opt, P_opt, P0_opt];
end

for i = 1:length(powers)
    % Skip direct estimation for anomalous points, keep global parameters
    if anomaly_flags_power(i) || anomaly_flags_voltage(i)
        continue;
    end
    
    window_size = dynamic_window_sizes(i);
    
    half_window = floor(window_size / 2);
    start_idx = max(1, i - half_window);
    end_idx = min(length(powers), i + half_window);
    
    if end_idx - start_idx + 1 < window_size
        start_idx = max(1, end_idx - window_size + 1);
    end
    
    % Extract window data
    V_window = voltages_pu(start_idx:end_idx);
    P_window = powers_pu(start_idx:end_idx);
    
    window_anomalies = anomaly_flags_power(start_idx:end_idx) | anomaly_flags_voltage(start_idx:end_idx);
    V_window_clean = V_window(~window_anomalies);
    P_window_clean = P_window(~window_anomalies);
    
    % Skip windows with insufficient valid data
    if length(V_window_clean) < 3 || any(isnan(V_window_clean)) || any(isnan(P_window_clean)) || all(P_window_clean < 0.01)
        continue;
    end
    
    window_obj_func = @(params) objective_function(params, V_window_clean, P_window_clean, V0);
    
    try
        [opt_result, ~] = fminsearch(window_obj_func, [Z_opt, I_opt, P0_opt], options);
        
        z_win = max(0, min(1, opt_result(1)));
        i_win = max(0, min(1, opt_result(2)));
        p_win = 1 - z_win - i_win;
        p0_win = max(0.1, opt_result(3));
        
        sum_win = z_win + i_win + p_win;
        z_win = z_win / sum_win;
        i_win = i_win / sum_win;
        p_win = p_win / sum_win;
        
        zip_full(i, :) = [z_win, i_win, p_win, p0_win];
    catch
    end
end

%% Apply local smoothing to reduce parameter oscillations
smoothing_window = 5;
for i = 1:length(powers)
    if anomaly_flags_power(i) || anomaly_flags_voltage(i)
        continue;
    end
    
    start_smooth = max(1, i - floor(smoothing_window/2));
    end_smooth = min(length(powers), i + floor(smoothing_window/2));
    
    smooth_range = start_smooth:end_smooth;
    valid_smooth = ~(anomaly_flags_power(smooth_range) | anomaly_flags_voltage(smooth_range));
    valid_indices = smooth_range(valid_smooth);
    
    if ~isempty(valid_indices)
        zip_full(i, :) = mean(zip_full(valid_indices, :), 1);
    end
end

%% Calculate power with the ZIP model for all points including anomalies
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

T = table(indices, voltages, powers, P_calc_all, P_calc_expression, ...
    zip_full(:,1), Z_calc_expression, ...
    zip_full(:,2), I_calc_expression, ...
    zip_full(:,3), P_param_expression, ...
    zip_full(:,4) * Pbase, ...
    window_sizes, ...
    anomaly_flags_voltage, anomaly_flags_power, ...
    'VariableNames', {'Index', 'Voltage', 'Power_measured', 'P_calc', 'P_calc_Formula', ...
                     'Z_coeff', 'Z_Component', ...
                     'I_coeff', 'I_Component', ...
                     'P_coeff', 'P_Component', ...
                     'P0_scaled', ...
                     'Window_Size', ...
                     'Voltage_Anomaly', 'Power_Anomaly'});
disp(T);

%% Plots
figure;
subplot(4,1,1);
plot(powers, 'b', 'LineWidth', 1.5, 'DisplayName', 'Measured Power');
hold on;
plot(P_calc_all, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Calculated Power (ZIP)');
% Mark anomalies
scatter(find(anomaly_flags_power), powers(anomaly_flags_power), 70, 'r', 'x', 'LineWidth', 2, 'DisplayName', 'Power Anomalies');
xlabel('Sample Index');
ylabel('Power (W)');
legend('Location', 'best');
title('M ltage (V)');
legend('Location', 'best');
title('Voltage Measurements with Anomalies');
grid on;

subplot(4,1,2);
plot(voltages, 'k-', 'LineWidth', 1.5, 'DisplayName', 'Voltage');
hold on;
% Mark voltage anomalies
scatter(find(anomaly_flags_voltage), voltages(anomaly_flags_voltage), 70, 'r', 'x', 'LineWidth', 2, 'DisplayName', 'Voltage Anomalies');
xlabel('Sample Index');
ylabel('Voltage (V)');
legend('Location', 'best');
title('Voltage Measurements with Anomalies');
grid on;

subplot(4,1,3);
plot(window_sizes, 'g-', 'LineWidth', 1.5);
xlabel('Sample Index');
ylabel('Window Size');
title('Dynamic Window Sizes (Larger Around Anomalies)');
grid on;

subplot(4,1,4);
bar(find(anomaly_flags_power | anomaly_flags_voltage), ones(sum(anomaly_flags_power | anomaly_flags_voltage), 1), 0.5, 'r');
ylim([0 1.5]);
xlabel('Sample Index');
title('Combined Anomaly Detection');
grid on;

%% Define the objective function here (nested function)
    function error = objective_function(params, voltages, powers, V0)
        % Extract parameters
        Z = params(1);
        I = params(2);
        P0 = params(3);
        
        % Calculate P coefficient to ensure sum is 1
        P = 1 - Z - I;
        
        % Handle invalid parameters
        if Z < 0 || I < 0 || P < 0 || P0 <= 0
            error = 1e10; % Large error for invalid solutions
            return;
        end
        
        % Calculate estimated power
        P_est = P0 * (Z * (voltages / V0).^2 + I * (voltages / V0) + P);
        
        % Mean squared error
        error = mean((P_est - powers).^2);
    end