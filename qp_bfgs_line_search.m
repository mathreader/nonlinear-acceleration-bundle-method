function [t_final] = qp_bfgs_line_search(A, b, x, p, c1, c2, max_line_search_iter)
%QP_BFGS_LINE_SEARCH Performs line search on the direction p starting at x.
%
% Inputs: 
%   A: (n * n * m) tensor, where A(:, :, i) represent A_i. Each A_i is
%       assumed to be positive semidefinite.
%   b: (n * m) vector, where b(:, i) represent b_i.  The sum of columns of
%       b_i is assumed to be the zero vector.
%   x: (n * 1) vector, as the input point for the iteration
%   p: (n * 1) vector, as the descent direction where we would like to 
%       perform line search on
%   c1, c2: scalar in [0, 1], with 0<c1<c2<1, being parameters for the
%       Armijo-Wolfe conditions.
%   max_line_search_iter: Maximum number of line search iterations.
%
% Goal: Find a step size t such that x+tp satisfy the Armijo-Wolfe
%   conditions
%
% Output:
%   t: scalar, representing the step size

alpha = 0;
beta = +inf;
t = 1;
num_iter = 0;
flag_done = 0;
t_final = 1;

while (num_iter < max_line_search_iter)
    
    [new_func_val, ~] = qp_function_eval(A, b, x+t*p);
    [old_func_val, ~] = qp_function_eval(A, b, x);

    [new_grad_val, new_grad_flag] = qp_gradient_oracle(A, b, x+t*p);
    [old_grad_val, old_grad_flag] = qp_gradient_oracle(A, b, x);

    h_val = new_func_val - old_func_val;
    h_grad = new_grad_val' * p;
    s = old_grad_val' * p;

    % Check Armijo condition A(t)
    if (h_val >= c1 * s * t)
        beta = t;
    % Check Wolfe condition W(t)
    elseif ((new_grad_flag == 1) || (old_grad_flag == 1) || (h_grad <= c2 * s))
        alpha = t;
    else
        if (flag_done == 0)
            fprintf('Line search terminates after %d iterations.\n', num_iter)
            t_final = t;
        end
        flag_done = 1;
    end
    
    if (beta < inf)
        t = (alpha + beta) / 2;
    else
        t = 2 * alpha;
    end
    
    num_iter = num_iter + 1;
end

if (flag_done == 0)
    fprintf('Line search does not terminate, output final value.\n')
    t_final = t;
end

end

