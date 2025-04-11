 %% ***************************************************************
%  filename: funV_1bit
%
%  Compute the gradient of the loss function f
%
%  F(U,V) = -sum P_Omega[log(Y.*f(UV') - (Y-1)/2)];  Laplacian noise /Logistic noise/ Gaussian noise
%
%  gradf_V = -P_Omega [gradf(UV')./(f(UV')+(Y-1)/2)]'*U 
%
%% *****************************************************************

function [Loss,gradV] = funV(U,y,ybar,nzidx,f,fprime,nr,nc,UVt)

UVnz = UVt(nzidx);

Loss = -sum(log(y.*f(UVnz) - ybar));

if nargout>=2
    
    B = zeros(nr,nc);
    
    B(nzidx) = -fprime(UVnz)./(f(UVnz)+ ybar);

    gradV = B'*U ;

end

end
