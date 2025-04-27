% Given voltage and power data
voltages = [1.0355; 1.0116; 0.9894; 1.0613; 1.0613; 1.0613; 1.0613; 1.0613; 1.0355; 1.0355; 1.0355; 1.0354; 1.0354; 1.0610; 1.0610; 1.0351; 1.0351; 1.0351; 1.0349; 1.0349; 1.0349];
powers = [0.1490; 0.1490; 0.1490; 0.1490; 0.1490; 0.1490; 0.1490; 0.1490; 0.1490; 0.1490; 0.1490; 0.1501; 0.1501; 0.1508; 0.1508; 0.1517; 0.1517; 0.1517; 0.1527; 0.1527; 0.1527];

% Initialize RLS parameters
lambda = 0.98;           % Forgetting factor
R = eye(3) * (10^-6);    % Measurement noise covariance
n_samples = length(voltages);

% Initialize state estimate and covariance matrix
x_hat = zeros(3, 1);    
P = eye(3) * 1000;       % Initial covariance matrix (large to indicate uncertainty)

V_0 = voltages(1);  
P_0 = powers(1);  

param_history = zeros(n_samples - 2, 3);

window_size = 3;

for k = window_size:n_samples
    V_window = voltages(k-window_size+1:k);
    P_window = powers(k-window_size+1:k);
    
    V_normalized = V_window / V_0;
    P_normalized = P_window / P_0;
    
    % Construct measurement matrix H_k for the current window
    H_k = [V_normalized.^2, V_normalized, ones(window_size, 1)];
    
    % Construct measurement vector b_k for the current window
    b_k = P_normalized;
    
    % Apply RLS equations for parameter estimation:
    P_inverse = (lambda^(-window_size)) * inv(P) + H_k' * inv(R) * H_k;
    P = inv(P_inverse);
    
    %Calculate Kalman gain
    K_k = P * H_k' * inv(R);
    
    % Update state estimate
    x_hat_new = x_hat + K_k * (b_k - H_k * x_hat);
    x_hat_new = x_hat_new / sum(x_hat_new);
    x_hat = x_hat_new;
    
    param_history(k-window_size+1, :) = x_hat';
end

fprintf('Estimated ZIP parameters at each time step:\n');
disp(array2table(param_history, 'VariableNames', {'Z_p', 'I_p', 'P_p'}));
