clc; clear; close all;

data1 = 'Data1.xlsx';
data = readtable(data1);
voltages = data.Vrms;
powers = data.P_ZIP;

%% Base values - adjusted based on your new values
Vbase = 240; % V
Pbase = 1500; % W

% Convert to per unit and ensure column vectors
voltages_pu = voltages / Vbase;
powers_pu = powers / Pbase;
voltages_pu = voltages_pu(:);
powers_pu = powers_pu(:);
voltages = voltages(:);
powers = powers(:);
V0 = 1.0;

%% Initial Least Squares ZIP Estimation

% Skip data points with zero or very small values to avoid division issues
valid_idx = powers_pu > 0.01;
valid_v = voltages_pu(valid_idx);
valid_p = powers_pu(valid_idx);

% Create regression matrix with valid data
H = [(valid_v / V0).^2, (valid_v / V0), ones(length(valid_v), 1)];

% using least square method for regularization
lambda = 0.001;
x = (H' * H + lambda * eye(3)) \ (H' * valid_p);

% Nominal power estimation
P0 = mean(valid_p);

% Extract ZIP coefficients
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

%% Dynamic Window Size Determination

% Calculate the rate of change for voltage and power
dv_dt = diff(voltages_pu);
dp_dt = diff(powers_pu);
dv_dt = [0; dv_dt];
dp_dt = [0; dp_dt];

% Calculate moving standard deviation of changes (indicator of volatility)
window_for_std = 3;

% calculating volatility with error checking
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

% Normalize volatility metrics
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

min_window_size = 3;
max_window_size = 10;

dynamic_window_sizes = round(max_window_size - (max_window_size - min_window_size) * combined_volatility);

dynamic_window_sizes = max(min_window_size, min(max_window_size, dynamic_window_sizes));

%% Adaptive ZIP Estimation Using Dynamic Window Sizing
P_calc_all = zeros(length(powers), 1);
zip_full = zeros(length(powers), 4); 

for i = 1:length(powers)
    zip_full(i, :) = [Z_opt, I_opt, P_opt, P0_opt];
end

% Process each point with its dynamic window
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
    
    if any(isnan(V_window)) || any(isnan(P_window)) || all(P_window < 0.01)
        continue;
    end
    
    window_obj_func = @(params) objective_function(params, V_window, P_window, V0);
    
    try
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
        
        % Store optimized parameters for this point
        zip_full(i, :) = [z_win, i_win, p_win, p0_win];
    catch
    end
end

%% Apply local smoothing to reduce parameter oscillations
smoothing_window = 3;
for i = 1:length(powers)
    start_smooth = max(1, i - floor(smoothing_window/2));
    end_smooth = min(length(powers), i + floor(smoothing_window/2));
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

T = table(indices, voltages, powers, P_calc_all, P_calc_expression, ...
    zip_full(:,1), Z_calc_expression, ...
    zip_full(:,2), I_calc_expression, ...
    zip_full(:,3), P_param_expression, ...
    zip_full(:,4) * Pbase, ...
    window_sizes, ...
    'VariableNames', {'Index', 'Voltage', 'Power_measured', 'P_calc', 'P_calc_Formula', ...
                     'Z_coeff', 'Z_Component', ...
                     'I_coeff', 'I_Component', ...
                     'P_coeff', 'P_Component', ...
                     'P0_scaled', ...
                     'Window_Size'});

disp(T);

%% Plots
figure;
subplot(3,1,1);
plot(powers, 'b', 'LineWidth', 1.5, 'DisplayName', 'Measured Power');
hold on;
plot(P_calc_all, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Calculated Power (ZIP)');
xlabel('Sample Index');
ylabel('Power (W)');
legend('Location', 'best');
title('Measured vs Estimated Power');
grid on;

subplot(3,1,2);
plot(voltages, 'k-', 'LineWidth', 1.5);
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