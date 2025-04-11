function p = Laplapdf(x,b)

%% Computes the pdf of a Gaussian distribution at specified values 

%% Compute values of PDF

 p = 0.5/b *exp(-abs(x)/b);
