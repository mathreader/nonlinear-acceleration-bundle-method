function [max_val, max_ind] = qp_function_eval(A, b, x)
%QP_FUNCTION_EVAL Function that evaluates the maximum of quadratic
%   functions.
%
% Inputs: 
%   A: (n * n * m) tensor, where A(:, :, i) represent A_i. Each A_i is
%       assumed to be positive semidefinite.
%   b: (n * m) vector, where b(:, i) represent b_i.  The sum of columns of
%       b_i is assumed to be the zero vector.
%   x: (n * 1) vector, as the input point where we would like to find the 
%       subgradient
%
% Goal: Solve the problem min_{x}(max_{1<=i<=n}(x' * A_i *x + b_i' * x))
%
% Output:
%   max_val: scalar value that is equal to max_{1<=i<=n}(x' * A_i *x + b_i' * x)
%   max_ind: index that achieves the maximum
%   

n = size(A, 1);
m = size(A, 3);

opt_list = zeros(m, 1);
for i = 1:m
    opt_list(i) = x' * A(:, :, i) * x / 2 + b(:, i)' * x;
end

[max_val, max_ind] = max(opt_list);

end

