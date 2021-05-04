function [g, flag_mult_max] = qp_gradient_oracle(A, b, x)
%QP_GRADIENT_ORACLE This function serves as an oracle for gradient
%  computation of the quadratic programming problem.
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
%   g: (n * 1) vector, representing one subgradient at x
%   flag_mult_max: a flag that indicates whether gradient is unique. 
%       Value is 1 if gradient is not unique (i.e. we have subgradients)
%       Value is 0 if gradient is unique (i.e. f is differentiable)

n = size(A, 1);
m = size(A, 3);

func_val = zeros(m, 1);

for i=1:m
    func_val(i) = x' * A(:, :, i) * x / 2+ b(:, i)' * x;
end

[~, max_ind] = max(func_val);

g = A(:, :, max_ind) * x + b(:, max_ind);


idx = find( func_val(:) == max(func_val(:)) );
if length(idx) > 1
    flag_mult_max = 1;
else
    flag_mult_max = 0;
end

    
end

