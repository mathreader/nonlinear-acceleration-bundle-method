function [x] = qp_baseline_cvx(A,b)
%QP_BASELINE_CVX This function solves the min-max problem of quadratic
%   objectives directly using cvx.
%
% Inputs: 
%   A: (n * n * m) tensor, where A(:, :, i) represent A_i. Each A_i is
%       assumed to be positive semidefinite.
%   b: (n * m) vector, where b(:, i) represent b_i.  The sum of columns of
%       b_i is assumed to be the zero vector.
%
% Goal: Solve the problem min_{x}(max_{1<=i<=n}(x' * A_i *x + b_i' * x))
%
% Output:
%   x: (n * 1) vector, representing the optimal solution to the 
%       optimization problem

n = size(A, 1);
m = size(A, 3);

cvx_begin
    variable z(n)
    y = z' * A(:, :, 1) * z / 2 + b(:, 1)' * z;
    for i = 2:m
        y = max(y, z' * A(:, :, i) * z / 2 + b(:, i)' * z);
    end
    minimize y
cvx_end

x = z;

end

