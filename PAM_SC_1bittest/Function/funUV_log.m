 %% ***************************************************************
%   filename: funUV_1bit
%
%  Compute the gradient of the loss function f
%
%  F(U,V) = -sum P_Omega[log(Y.*f(UV') - (Y-1)/2)];  Laplacian noise /Logistic noise/ Gaussian noise
%
%  gradf_U = -P_Omega [gradf(UV')./(f(UV')+(Y-1)/2)]*V 
%
%  gradf_V = -P_Omega [gradf(UV')./(f(UV')+(Y-1)/2)]'*U
%% *********************************************************************

function  [Loss,gradU,gradV] = funUV_log(U,V,y,ybar,spidx,f,nr,nc,UVt)         

UVnz = UVt(spidx);

Loss = -sum(log(y.*f(UVnz) - ybar));

if nargout>1
    
    B = zeros(nr,nc);

    B(spidx) = -y./(1+exp(y.*UVnz));
    
    gradU = B*V ;  

end

if nargout>2

  gradV = B'*U;

end