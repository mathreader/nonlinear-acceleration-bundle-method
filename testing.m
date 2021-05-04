%% Initialization

% Define size of the problem
% n is dimension of each matrix, m is the number of matrices.
n = 100;
m = 20;

% Generate positive semidefinite A
A_temp = randn(n, n, m);
A = zeros(n, n, m);
for i = 1:m
    A(:, :, i) = A_temp(:, :, i)' * A_temp(:, :, i);
end


%A = [2 0; 0 1];

% Generate b with columns summing to 0.
b = zeros(n, m);

for i = 1:(m-1)
    temp = randn(n, 1);
    b(:, i) = temp;
    b(:, m) = b(:, m) - temp;
end

%b = zeros(n, m);

%% Method 0: Baseline directly using CVX

tic

x_baseline = qp_baseline_cvx(A, b);

[opt_baseline, ~] = qp_function_eval(A, b, x_baseline);

t0 = toc;

%% Method 1: Prox-descent method

tic

% Additional parameters
x0 = rand(n, 1);
eps = 1e-10;
max_iter = 100;

% Finding a suitable lambda for convergence
K_mat = zeros(m, 1);
for i=1:m
    K_mat(i) = norm(A(:, :, i));
end

K_norm = norm(K_mat);
lambda = 2 * K_norm;

% Running the algorithm
[x_prox_descent, iter_prox_descent, x_list_prox_descent, opt_list_prox_descent] ...
    = qp_prox_descent(A, b, x0, lambda, eps, max_iter);

t1 = toc;

%% Method 2: Proximal Bundle Method

tic

% Additional parameters
x0 = ones(n, 1);
eps = 1e-10;
max_iter = 100;

lambda = 100;
beta = 0.5;

% Running the algorithm
[x_list_bundle, z_list_bundle, opt_list_bundle, step_type_list_bundle] ...
    = qp_proximal_bundle(A, b, x0, lambda, beta, eps, max_iter);


t2 = toc;

%% Method 3: BFGS Method

tic

% Additional parameters
B0 = eye(n);
x0 = rand(n, 1);
c1 = 0.25;
c2 = 0.5;
eps = 1e-10;
max_bfgs_iter = 100;
max_line_search_iter = 20;

[x_list_bfgs, opt_list_bfgs] = qp_bfgs(A, b, B0, x0, c1, c2, eps, max_bfgs_iter, max_line_search_iter);

x_list_bfgs(:, end)
qp_function_eval(A, b, x_list_bfgs(:, end))


t3 = toc;

%% Plotting

opt_val = 0;

%plot(1:size(opt_list_prox_descent), log(opt_list_prox_descent(1:end) - opt_val))
%plot(11:size(opt_list_prox_descent), log(opt_list_prox_descent(11:end) - opt_val))
%plot(11:(size(opt_list_bundle)-1), log(opt_list_bundle(12:end) - opt_val))
plot(11:(size(opt_list_bfgs)), log(opt_list_bfgs(11:end) - opt_val))