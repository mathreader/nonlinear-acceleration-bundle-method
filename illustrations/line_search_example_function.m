function [result] = line_search_example_function(u, v)
%LINE_SEARCH_EXAMPLE_FUNCTION Summary of this function goes here
%   Detailed explanation goes here

if (u >= norm(v) / 2)
    result = sqrt(u^2 + 2* v^2);
else
    result = (u + 4 * abs(v)) / 3;
end


end

