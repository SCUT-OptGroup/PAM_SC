%% Computes the negative log-likelihood function and its gradient

function [grad,loss] = funX_grad(x,y,ybar,nzidx,f,fprime,nr,nc)

grad = zeros(nr,nc);

grad(nzidx) = -fprime(x)./(f(x)+ ybar);

if nargout > 1
    
    loss = -sum(log(y.*f(x) - ybar));

end