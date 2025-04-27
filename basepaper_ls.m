V_measured =  [1.0355; 1.0116; 0.9894; 1.0613; 1.0613; 1.0613; 1.0613; 1.0613; 1.0355; 1.0355; 1.0355; 1.0354; 1.0354; 1.0610; 1.0610; 1.0351; 1.0351; 1.0351; 1.0349; 1.0349; 1.0349];
P_measured =  [0.1490; 0.1490; 0.1490; 0.1490; 0.1490; 0.1490; 0.1490; 0.1490; 0.1490; 0.1490; 0.1490; 0.1501; 0.1501; 0.1508; 0.1508; 0.1517; 0.1517; 0.1517; 0.1527;0.1527; 0.1527]; 

num_readings = length(V_measured);
group_size = 3;
num_groups = floor(num_readings / group_size);

V_groups = reshape(V_measured(1:num_groups * group_size), group_size, [])';
P_groups = reshape(P_measured(1:num_groups * group_size), group_size, [])';

function [Z_final, I_final, P_final] = calculate_zip_parameters(V_group, P_group)
    V_nominal = mean(V_group);
    P_nominal = mean(P_group);
    V_normalized = V_group / V_nominal;
    P_normalized = P_group / P_nominal;

    % Construct the design matrix A and target vector b
    A = [(V_normalized.^2)', V_normalized', ones(length(V_normalized), 1)];
    b = P_normalized';

    % Solve for ZIP coefficients using Least Squares method
    x = (A' * A) \ (A' * b); 

    Z_new = max(0, x(1)); 
    I_new = max(0, x(2));
    P_new = max(0, x(3));

    total_sum = Z_new + I_new + P_new;
    Z_final = Z_new / total_sum;
    I_final = I_new / total_sum;
    P_final = P_new / total_sum;

    if any(isnan([Z_final, I_final, P_final]))
        Z_final = NaN;
        I_final = NaN;
        P_final = NaN;
    end
end

% Calculate ZIP parameters for each group
zip_parameters = zeros(num_groups, 3);
previous_values = [NaN NaN NaN];

for i = 1:num_groups
    [Z_final, I_final, P_final] = calculate_zip_parameters(V_groups(i,:), P_groups(i,:));
    
    if any(isnan([Z_final, I_final, P_final]))
        zip_parameters(i,:) = previous_values;
    else
        zip_parameters(i,:) = [Z_final, I_final, P_final];
        previous_values = [Z_final, I_final, P_final];
    end
end

disp('Index | Voltage  | Power    | Impedance (Z) | Current (I) | Power (P)');
disp('---------------------------------------------------------------');

zip_parameters_expanded = repelem(zip_parameters, group_size, 1); 

for i = 1:num_groups * group_size
    fprintf('%5d  | %.4f  | %.4f  | %.4f         | %.4f       | %.4f\n', ...
        i, V_measured(i), P_measured(i), zip_parameters_expanded(i, :));
end