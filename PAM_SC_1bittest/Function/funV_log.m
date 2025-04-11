 %% ***************************************************************
%  filename: funV_log
%
%  Compute the gradient of the loss function f
%
%  F(U,V) = -sum P_Omega[log(Y.*f(UV') - (Y-1)/2)];  Laplacian noise /Logistic noise/ Gaussian noise
%
%  gradf_V = -P_Omega [gradf(UV')./(f(UV')+(Y-1)/2)]'*U 
%
%% *****************************************************************

function [Loss,gradV] = funV_log(U,y,ybar,spidx,f,nr,nc,UVt)           

UVnz = UVt(spidx);

Loss = -sum(log(y.*f(UVnz) - ybar));

if nargout>=2

    B = zeros(nr,nc);

    B(spidx) = -y./(1+exp(y.*UVnz));

    gradV = B'*U ;

end

end
