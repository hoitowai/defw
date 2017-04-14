function x = proj_l1(y,r)
% this function computes the projection of vector y into the l1 ball with radius r
n = length(y);
y_sort = sort(abs(y),'descend');

if sum(y_sort) <= r
    x = y;
else
    for i = 1:n
        lam_try = (sum( y_sort(1:i) ) - r )/i; 
        if y_sort(i) - lam_try < 0
            % THE PREVIOUS CASE WAS GOOD, take it back
            lam_try = (sum( y_sort(1:(i-1)) ) - r )/(i-1);
            x = max(0,abs(y)-lam_try).*sign(y);
            break;
        end
        if i == n
            x = max(0,abs(y)-lam_try).*sign(y);
        end
    end
end