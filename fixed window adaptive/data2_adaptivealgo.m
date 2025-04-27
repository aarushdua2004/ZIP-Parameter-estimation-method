clc; clear; close all;

data2 = 'Data2.xlsx';
data = readtable(data2);
voltages = data.Vrms;
powers = data.P_ZIP;

%% Base values - adjusted based on your new values
Vbase = 400; % V
Pbase = 1000; % W

voltages_pu = voltages / Vbase;
powers_pu = powers / Pbase;

%column vector conversion for proper matrix use
voltages_pu = voltages_pu(:);
powers_pu = powers_pu(:);
voltages = voltages(:);
powers = powers(:);
V0 = 1.0;

%% Initial Least Squares ZIP Estimation

% skipping redundant data
valid_idx = powers_pu > 0.01;
valid_v = voltages_pu(valid_idx);
valid_p = powers_pu(valid_idx);

% Create regression matrix with valid data
H = [(valid_v / V0).^2, (valid_v / V0), ones(length(valid_v), 1)];

% Use least squares with regularization
lambda = 0.001; % Small regularization factor
x = (H' * H + lambda * eye(3)) \ (H' * valid_p);

P0 = mean(valid_p);

%initial parameter estimation
Z_initial = x(1) / P0;
I_initial = x(2) / P0;
P_initial = x(3) / P0;

Z_initial = max(0, Z_initial);
I_initial = max(0, I_initial);
P_initial = max(0, P_initial);

% normalizing wrt condition that sum should be 1
sum_zip = Z_initial + I_initial + P_initial;
if sum_zip > 0
    Z_initial = Z_initial / sum_zip;
    I_initial = I_initial / sum_zip;
    P_initial = P_initial / sum_zip;
else
    %if fail, set hardcoded ZIP
    Z_initial = 0.3;
    I_initial = 0.3;
    P_initial = 0.4;
end

fprintf('Initial ZIP values: Z=%.4f, I=%.4f, P=%.4f\n', Z_initial, I_initial, P_initial);

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

%% Optimization for global parameters
options = optimset('Display', 'off', 'MaxIter', 1000);

function_handle = @(params) objective_function(params, voltages_pu, powers_pu, V0);

initial_guess = [Z_initial, I_initial, P0];

% Run optimization
[optimized_params, ~] = fminsearch(function_handle, initial_guess, options);

%extract optimized parameters
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

%% Adaptive ZIP Estimation Using Sliding Window
window_size = 10;
num_windows = length(voltages) - window_size + 1;
best_zip_values = zeros(num_windows, 4); % [Z, I, P, P0]
P_calc_all = zeros(length(powers), 1);

for i = 1:num_windows
    % Extract window data
    V_window = voltages_pu(i:i+window_size-1);
    P_window = powers_pu(i:i+window_size-1);
    
    if any(isnan(V_window)) || any(isnan(P_window)) || all(P_window < 0.01)
        % Use global parameters if window data is problematic
        best_zip_values(i, :) = [Z_opt, I_opt, P_opt, P0_opt];
        continue;
    end
    
    % Define objective function for this window
    window_obj_func = @(params) objective_function(params, V_window, P_window, V0);
    
    init_params = [Z_opt, I_opt, P0_opt];
    
    try
        [opt_result, ~] = fminsearch(window_obj_func, init_params, options);
        
        z_win = max(0, min(1, opt_result(1)));
        i_win = max(0, min(1, opt_result(2)));
        p_win = 1 - z_win - i_win;
        p0_win = opt_result(3);
        
        p0_win = max(0.1, p0_win);
        
        sum_win = z_win + i_win + p_win;
        z_win = z_win / sum_win;
        i_win = i_win / sum_win;
        p_win = p_win / sum_win;
        
        best_zip_values(i, :) = [z_win, i_win, p_win, p0_win];
    catch
        best_zip_values(i, :) = [Z_opt, I_opt, P_opt, P0_opt];
    end
end

smooth_zip = best_zip_values;

%% Fill in ZIP values for all time points and calculate power
zip_full = zeros(length(powers), 4);

% First points (before sliding window)
for i = 1:window_size-1
    zip_full(i, :) = smooth_zip(1, :);
end

% Middle points (covered by sliding window)
for i = 1:num_windows
    zip_full(i+window_size-1, :) = smooth_zip(i, :);
end

% End points (after sliding window)
for i = (num_windows+window_size):length(powers)
    if i <= length(powers)  % Add check to prevent index out of bounds
        zip_full(i, :) = smooth_zip(end, :);
    end
end

% Calculate power with the ZIP model
for i = 1:length(powers)
    Z_val = zip_full(i, 1);
    I_val = zip_full(i, 2);
    P_val = zip_full(i, 3);
    P0_val = zip_full(i, 4);
    
    v_ratio = voltages_pu(i) / V0;
    P_calc_all(i) = P0_val * (Z_val * v_ratio^2 + I_val * v_ratio + P_val) * Pbase;
end

%% Calculate individual components
Z_component = zip_full(:,1) .* (voltages_pu/V0).^2 .* zip_full(:,4) * Pbase;
I_component = zip_full(:,2) .* (voltages_pu/V0) .* zip_full(:,4) * Pbase;
P_component = zip_full(:,3) .* zip_full(:,4) * Pbase;

%% Create calculation expressions
P_calc_expression = strings(length(powers), 1);
for i = 1:length(powers)
    % Power calculation expression
    P_calc_expression(i) = sprintf('(%.4f*(%.3f/%.3f)^2 + %.4f*(%.3f/%.3f) + %.4f)*%.2f*%.0f', ...
        zip_full(i,1), voltages_pu(i), V0, ...
        zip_full(i,2), voltages_pu(i), V0, ...
        zip_full(i,3), zip_full(i,4), Pbase);
end

%% Final Table Output
T = table((1:length(powers))', voltages, powers, P_calc_all, P_calc_expression, ...
    zip_full(:,1), ...
    zip_full(:,2), ...
    zip_full(:,3), ...
    zip_full(:,4) * Pbase, ...
    'VariableNames', {'Index', 'Voltage', 'Power_measured', 'P_calc', 'P_calc_Formula', ...
                     'Z_coeff',  ...
                     'I_coeff', ...
                     'P_coeff',  ...
                     'P0_scaled'});

disp(T);

%% Calculate error metrics
RMSE = sqrt(mean((powers - P_calc_all).^2));
MAE = mean(abs(powers - P_calc_all));
MAPE = mean(abs((powers - P_calc_all) ./ powers)) * 100;

fprintf('RMSE: %.2f W\n', RMSE);
fprintf('MAE: %.2f W\n', MAE);
fprintf('MAPE: %.2f%%\n', MAPE);

%% Plots
figure;
subplot(2,1,1);
plot(powers, 'b', 'LineWidth', 1.5, 'DisplayName', 'Measured Power');
hold on;
plot(P_calc_all, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Calculated Power (ZIP)');
xlabel('Sample Index');
ylabel('Power (W)');
legend('Location', 'best');
title('Measured vs Estimated Power');
grid on;

subplot(2,1,2);
plot(voltages, 'k-', 'LineWidth', 1.5);
xlabel('Sample Index');
ylabel('Voltage (V)');
title('Voltage Measurements');
grid on;

figure;
plot(zip_full(:,1), 'r-', 'LineWidth', 1.5, 'DisplayName', 'Z coefficient');
hold on;
plot(zip_full(:,2), 'g-', 'LineWidth', 1.5, 'DisplayName', 'I coefficient');
plot(zip_full(:,3), 'b-', 'LineWidth', 1.5, 'DisplayName', 'P coefficient');
xlabel('Sample Index');
ylabel('Coefficient Value');
legend('Location', 'best');
title('ZIP Coefficients');
grid on;