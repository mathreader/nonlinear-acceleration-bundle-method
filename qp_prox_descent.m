function [x_curr, current_iter, x_list, opt_list] = qp_prox_descent(A, b, x0, lambda, eps, max_iter)
%QP_PROX_DESCENT This function solves the min-max problem of quadratic
%   objectives using the prox-descent method.
%
% Inputs: 
%   A: (n * n * m) tensor, where A(:, :, i) represent A_i. Each A_i is
%       assumed to be positive semidefinite.
%   b: (n * m) vector, where b(:, i) represent b_i.  The sum of columns of
%       b_i is assumed to be the zero vector.
%   x0: (n * 1) vector, representing starting point.
%   lambda: scalar, regularization parameter
%   eps: error where we stop the iteration
%   max_iter: maximum number of iterations to be run
%
% Goal: Solve the problem min_{x}(max_{1<=i<=n}(x' * A_i *x + b_i' * x))
%
% Output:
%   x: (n * 1) vector, representing the optimal solution to the 
%       optimization problem
%   current_iter: iterations used by the method
%   x_list: list of iterates x_k obtained in the iterations
%   opt_list: list of optimal values f(x_k) obtained in the iterations

n = size(A, 1);
m = size(A, 3);

x_prev = x0;
x_curr = x0;
x_list = zeros(n, max_iter);
opt_list = zeros(max_iter, 1);

for current_iter=1:max_iter
    
    cvx_begin
        variable z(n)
        variable t
        expression con_lower_bound(m)
        
        rep_t = repmat(t, m, 1);
        for i = 1:m
            f_val = x_curr' * A(:, :, i) * x_curr / 2 + b(:, i)' * x_curr;
            grad_val = A(:, :, i) * x_curr + b(:, i);
            con_lower_bound(i) = f_val + grad_val' * (z-x_curr);
        end
        
        y = t + lambda * sum_square(z-x_curr) / 2;
        minimize y
        subject to 
            rep_t >= con_lower_bound;
    cvx_end
    
    
    x_prev = x_curr;
    x_curr = z;
    x_list(:, current_iter) = z;
    opt_list(current_iter) = y;
    
    if norm(x_curr - x_prev) <= eps
        break
    end
end

x_list = x_list(:, 1:current_iter);
opt_list = opt_list(1:current_iter);

end

