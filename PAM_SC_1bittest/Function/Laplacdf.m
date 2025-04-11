function p = Laplacdf(x,b)

%% Computes the cdf of a Laplacian distribution at specified values 

%% Compute values of CDF
p=x;

indx = x<0;

p(indx) = 0.5 *exp(x(indx)/b);

p(~indx) = 1-0.5*exp(-(x(~indx))/b);

end
