%% Plot the line search example function

lbound = -10;
hbound = 10;
num_points = 21;

s1 = linspace(lbound, hbound, num_points);
s2 = linspace(lbound, hbound, num_points);

Z = zeros(num_points, num_points);

for i=1:1:num_points
    for j=1:1:num_points
        Z(i, j) = line_search_example_function(s1(i), s2(j));
    end
end

[X,Y] = meshgrid(s1, s2); 

surf(X, Y, Z)
xlabel('v')
ylabel('u')

%% Perform subgradient method on this function


num_iter = 100;
x0 = [2, -1];
upper_bound_search = 10000;


x = x0;
x_list = zeros(num_iter+1, 2);
value_list = zeros(num_iter+1, 2);


x_list(1, :) = x0;
value_list(1) = line_search_example_function(x(1), x(2));
index = 1;

for i=1:1:num_iter
    if (x(1) > abs(x(2)) / 2)
        grad = [x(1) / sqrt(x(1)^2 + 2* x(2)^2), 2 * x(2) / sqrt(x(1)^2 + 2* x(2)^2)];
    else
        grad = [1/3, 4/3 * sign(x(2))];
    end
    line_search_prob = @(a) line_search_example_function(x(1)-a*grad(1), x(2)-a*grad(2));
    a = fminbnd(line_search_prob, 0, upper_bound_search);
    x = x - a * grad;
    x_list(i+1, :) = x;
    value_list(i+1) = line_search_example_function(x(1), x(2));
end

x_plot = x_list(:, 1)';
y_plot = x_list(:, 2)';

plot(x_plot(1:10), y_plot(1:10), '-rs', 'MarkerSize',2)
xlim([-3 3])
ylim([-3 3])
xL = xlim;
yL = ylim;
x_axis = line([0 0], yL);  %x-axis
y_axis = line(xL, [0 0]);  %y-axis
x_axis.Color = 'black';
y_axis.Color = 'black';

