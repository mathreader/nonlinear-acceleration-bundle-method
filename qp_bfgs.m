function [x_list, opt_list] = qp_bfgs(A, b, B0, x0, c1, c2, eps, max_bfgs_iter, max_line_search_iter)
%QP_BFGS BFGS method applied on the quadratic programming problem
%
% Inputs: 
%   A: (n * n * m) tensor, where A(:, :, i) represent A_i. Each A_i is
%       assumed to be positive semidefinite.
%   b: (n * m) vector, where b(:, i) represent b_i.  The sum of columns of
%       b_i is assumed to be the zero vector.
%   B0: (n * n) matrix, representing initial Hessian.
%   x0: (n * 1) vector, representing starting point.
%   c1, c2: scalar in [0, 1], with 0<c1<c2<1, being parameters for the
%       Armijo-Wolfe conditions.
%   eps: error where we stop the iteration
%   max_bfgs_iter: maximum number of iterations for bfgs
%   max_line_search_iter: maximum number of iterations for line search
%
% Goal: Solve the problem min_{x}(max_{1<=i<=n}(x' * A_i *x/2 + b_i' * x))
%
% Output:
%   x: (n * 1) vector, representing the optimal solution to the 
%       optimization problem

n = size(A, 1);
m = size(A, 3);

x_prev = x0;
x_curr = x0;
B = B0;

x_list = zeros(n, max_bfgs_iter);
opt_list = zeros(max_bfgs_iter, 1);

for current_iter=1:max_bfgs_iter
    
    % Generate descent direction
    [current_grad, ~] = qp_gradient_oracle(A, b, x_curr);
    p = B \ (-current_grad);
    
    % Find step size for line search 
    line_step_size = qp_bfgs_line_search(A, b, x_curr, p, c1, c2, max_line_search_iter);
    
    % Compute next iterate
    s = line_step_size * p;
    x_prev = x_curr;
    x_curr = x_curr + s;
    
    % Compute next approximate Hessian
    [grad_prev, ~] = qp_gradient_oracle(A, b, x_prev);
    [grad_curr, ~] = qp_gradient_oracle(A, b, x_curr);
    y = grad_curr - grad_prev;
    B = B + (y * y') / (y' * s) - (B * s * s' * B') / (s' * B * s);
    
    x_list(:, current_iter) = x_curr;
    opt_list(current_iter) = qp_function_eval(A, b, x_curr);
    
    if (norm(x_curr - x_prev) < eps)
        break
    end   
end

x_list = x_list(:, 1:current_iter);
opt_list = opt_list(1:current_iter);

end

