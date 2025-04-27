voltages = [402, 402, 402, 402, 402, 402, 402,402, 402, 402, 402, 402, 402, 402, 402,402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402,402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402,402, 402, 402, 402, 402, 402, 402, 401, 401, 402, 401, 401, 402, 402, 402, 402, 401, 402, 401, 401, 402, 402, 402, 402, 401, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402,402, 402, 401, 401, 401, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 402, 401, 400, 385, 360, 352, 352, 352, 352, 342, 319, 305, 305, 305, 305,305,305, 305, 305, 305, 305, 310, 341, 355, 355, 355, 355, 355, 368, 397, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405,405, 405, 405, 405, 405, 405, 405, 402, 368, 351, 313, 303, 303, 303, 303, 303, 303, 303, 302, 302, 302, 302, 303, 313, 348, 353, 353, 353, 353, 354, 353, 354, 354, 355, 373, 401, 404, 411, 417, 416, 405, 402, 400, 399, 399, 399, 399, 399, 399, 399, 399, 399, 405, 405, 405, 405, 405, 404, 405, 405, 405, 405, 405, 405, 404, 404, 404, 405, 405, 405, 405, 405, 404, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405, 405];
powers = [1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160,0, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160,1800, 1160, 1160, 1160, 1220, 1360, 1360, 1360, 1360, 1360, 1360, 1360, 1360, 1360,1360, 1360, 1360, 1360, 1360, 1360, 1360, 1360, 1360, 1360, 1360, 1360, 1360, 1470, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560,0, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1560, 1370, 1360, 1360, 1350, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160,0,0, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1150, 1120, 1090, 1090, 1090, 1090, 1090, 1070, 1060, 1060, 1060, 1060, 1060, 1060, 1060, 1060, 1060, 1060, 1080,1500, 1090, 1090, 1090, 1090, 1090, 1100, 1140, 1160, 1170, 1170, 1170, 1170, 1170, 1170, 1170, 1170, 1170, 1170, 1170, 1170, 1170, 1170, 1170, 1170, 1170, 1170, 1170, 1170, 1170, 1170, 1170, 1140, 1090, 1080, 1050, 1060, 1060, 1060, 1150, 1260, 1260, 1260, 1260, 1260, 1260, 1260, 1260, 1270, 1290, 1300, 1300, 1300, 1300, 1300, 1300, 1250, 1090, 1090, 1110, 1140, 1160, 1170, 1190, 1180, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1160, 1130, 959, 959, 959, 959, 959, 959, 959, 959, 959, 959, 959, 959, 959, 959, 959, 959, 959, 959, 959];

voltages = voltages(:);
powers = powers(:);
indices = 1:length(powers);
window_size = 3;  
percentage_threshold = 0.15;  
step_size = 1;  

% Initialize arrays to store results
expected_power = zeros(size(powers));
high_threshold = zeros(size(powers));
low_threshold = zeros(size(powers));
is_normal = true(size(powers));

for i = 1:step_size:length(powers)
    start_idx = max(1, i - floor(window_size/2));
    end_idx = min(length(powers), i + floor(window_size/2));
    
    window_indices = start_idx:end_idx;
    window_voltages = voltages(window_indices);
    window_powers = powers(window_indices);
    
    % Perform linear regression on window data
    X = [ones(length(window_voltages), 1), window_voltages];
    beta = X \ window_powers;
    
    expected_power(i) = beta(1) + beta(2) * voltages(i);
    
    high_threshold(i) = expected_power(i) * (1 + percentage_threshold);
    low_threshold(i) = expected_power(i) * (1 - percentage_threshold);
    
    % Determine if the center point is an anomaly
    if powers(i) > high_threshold(i) || powers(i) < low_threshold(i)
        is_normal(i) = false;
    end
end

filtered_indices = indices(is_normal);
filtered_voltages = voltages(is_normal);
filtered_powers = powers(is_normal);

anomaly_count = sum(~is_normal);
percentage_anomalies = (anomaly_count / length(powers)) * 100;

fprintf('Total data points: %d\n', length(powers));
fprintf('Identified anomalies: %d (%.2f%%)\n', anomaly_count, percentage_anomalies);
fprintf('Remaining normal data points: %d\n', sum(is_normal));
fprintf('Window size: %d data points\n', window_size);
fprintf('Percentage threshold: ±%.1f%% of expected power value\n', percentage_threshold * 100);

figure('Position', [100, 100, 1200, 800]);

subplot(2, 1, 1);
scatter(voltages, powers, 30, 'b', 'filled', 'MarkerFaceAlpha', 0.5);
hold on;
scatter(voltages(~is_normal), powers(~is_normal), 50, 'r', 'x', 'LineWidth', 2);
% Plot the expected power curve (connecting all expected points)
plot(voltages, expected_power, 'g-', 'LineWidth', 2);
% Plot the threshold curves
plot(voltages, high_threshold, 'r--', 'LineWidth', 1.5);
plot(voltages, low_threshold, 'r--', 'LineWidth', 1.5);
xlabel('Voltage');
ylabel('Power');
title('Windowed Power Anomaly Detection');
legend('Original Data', 'Identified Anomalies', 'Expected Power', 'Threshold Boundaries');
grid on;

subplot(2, 1, 2);
plot(indices, powers, 'b-', 'LineWidth', 1);
hold on;
plot(indices, expected_power, 'g-', 'LineWidth', 2);
plot(indices, high_threshold, 'r--', 'LineWidth', 1.5);
plot(indices, low_threshold, 'r--', 'LineWidth', 1.5);
scatter(indices(~is_normal), powers(~is_normal), 50, 'r', 'x', 'LineWidth', 2);
xlabel('Sample Index');
ylabel('Power (W)');
title('Power Time Series with Windowed Thresholds');
legend('Measured Power', 'Expected Power', 'High Threshold', 'Low Threshold', 'Anomalies');
grid on;

figure('Position', [100, 100, 1200, 500]);
plot(indices, powers, 'b-', 'LineWidth', 1, 'DisplayName', 'Original Power');
hold on;
plot(filtered_indices, filtered_powers, 'g-', 'LineWidth', 2, 'DisplayName', 'Filtered Power');
xlabel('Sample Index');
ylabel('Power (W)');
title(sprintf('Original vs. Filtered Power Data (%d anomalies removed)', anomaly_count));
legend('Location', 'best');
grid on;

if ~isempty(filtered_powers)
    X_filtered = [ones(length(filtered_voltages), 1), filtered_voltages];
    beta_filtered = X_filtered \ filtered_powers;
    fprintf('Regression on filtered data: Power = %.4f + %.4f * Voltage\n', beta_filtered(1), beta_filtered(2));
end

filtered_data = [filtered_voltages, filtered_powers];