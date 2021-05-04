function [x_list, z_list, opt_list, step_type_list] = qp_proximal_bundle(A, b, x0, lambda, beta, eps, max_iter)
%QP_PROXIMAL_BUNDLE This function solves the min-max problem of quadratic
%   objectives using the proximal bundle method.
%
% Inputs: 
%   A: (n * n * m) tensor, where A(:, :, i) represent A_i. Each A_i is
%       assumed to be positive semidefinite.
%   b: (n * m) vector, where b(:, i) represent b_i.  The sum of columns of
%       b_i is assumed to be the zero vector.
%   x0: (n * 1) vector, representing starting point.
%   lambda: scalar, regularization parameter for proximal update
%   beta: scalar, descent parameter (within 0 and 1)
%   eps: error where we stop the iteration
%   max_iter: maximum number of iterations
%
% Goal: Solve the problem min_{x}(max_{1<=i<=n}(x' * A_i *x + b_i' * x))
%
% Output:
%   x: (n * 1) vector, representing the optimal solution to the 
%       optimization problem

% TODO

n = size(A, 1);
m = size(A, 3);

x = x0;
z = x0;
[g, ~] = qp_gradient_oracle(A, b, z);

x_list = zeros(n, max_iter+1);
z_list = zeros(n, max_iter+1);
g_list = zeros(n, max_iter+1);
opt_list = zeros(max_iter+1, 1);
step_type_list = zeros(max_iter, 1);

x_list(:,1) = x;
z_list(:,1) = z;
g_list(:,1) = g;

for current_iter=1:max_iter
    
    % Compute the proximal update via quadratic programming
    cvx_begin
        variable w(n, 1)
        variable t
        expression con_lower_bound(current_iter)
        
        rep_t = repmat(t, current_iter, 1);
        for i = 1:current_iter
            [f_val_z, ~] = qp_function_eval(A, b, z_list(:, i));
            con_lower_bound(i) = f_val_z + g_list(:, i)' * (w-z_list(:, i));
        end
        
        y = t + lambda * sum_square(w-x_list(:,current_iter)) / 2;
        minimize y
        subject to 
            rep_t >= con_lower_bound;
    cvx_end
    
    % Update z_{k+1} into the storage
    z_list(:, current_iter + 1) = w;
    
    
    % Compute F(x_{k}) and F(z_{k})
    [f_val_x, ~] = qp_function_eval(A, b, x_list(:, current_iter));
    [f_val_z, ~] = qp_function_eval(A, b, z_list(:, current_iter + 1));
    
    % Compute \tilde{F}^k(z_{k+1})
    f_cutting_plane_z_list = zeros(current_iter, 1);
    for j = 1:current_iter
        [f_val_new_z, ~] = qp_function_eval(A, b, z_list(:, j));
        f_cutting_plane_z_list(j) = f_val_new_z ...
            + g_list(:, j)' * (z_list(:, current_iter + 1) - z_list(:, j));
    end
    f_cutting_plane_z = max(f_cutting_plane_z_list);
    
    % Check epsilon-optimality
    if (abs(f_val_x - f_cutting_plane_z) <= eps)
        % If already optimal, jump out of loop
        break;
    else
        % Decide between serious and null step
        if (f_val_z <= f_val_x - beta * (f_val_x - f_cutting_plane_z))
            % Serious step
            x_list(:, current_iter + 1) = z_list(:, current_iter + 1);
            step_type_list(current_iter) = 1;
        else
            % Null step
            x_list(:, current_iter + 1) = x_list(:, current_iter);
            step_type_list(current_iter) = -1;
        end
        
        [opt_list(current_iter + 1), ~] = qp_function_eval(A, b, x_list(:, current_iter + 1));
        [g_list(:, current_iter + 1), ~] = qp_gradient_oracle(A, b, z_list(:, current_iter + 1));
    end
end

x_list = x_list(:, 1:current_iter);
z_list = z_list(:, 1:current_iter);
opt_list = opt_list(1:current_iter);
step_type_list =step_type_list(1:current_iter);

end

