%% ********************************************************************
%  filename: SCM_AMM
%% **********************************************************************
%% Majorized alternating proximal minimization method for solving
%  
%  min{ F(U,V) + 0.5*mu(||U||_F^2+||V||_F^2)}+lambda(||U||_{2,0}+||V||_{2,0}) (*)
%  
%  where F(U,V) = -sum P_Omega[log(Y.*f(UV') - (Y-1)/2)]; f is Laplacian noise /Logistic noise/ Gaussian noise
% 
%  in the order: U--->V---->U
%% **************************************************************************
%%  2024-04-16
%% *************************************************************************

function [rankX,X,obj,iter] = SC_PAMM(Mstar,Xold,Zk,P,Q,d,spidx,lambda,mu,pars,fun_grad,OPTIONS)

if isfield(OPTIONS,'printyes');   printyes  = OPTIONS.printyes;    end
if isfield(OPTIONS,'Xmaxiter');    maxiter   = OPTIONS.Xmaxiter;     end
if isfield(OPTIONS,'Xtol');        tol       = OPTIONS.Xtol;         end
if isfield(OPTIONS,'objtol');     objtol    = OPTIONS.objtol;        end
if isfield(OPTIONS,'Lip_const');  Lip      = OPTIONS.Lip_const;    end

eta = Lip;  q = pars.q;

lgamma = 1.0e-8;

gratio = 0.8;

gamma1 = 1.0e-2;   gamma2 = 1.0e-2;

gam1_mu = gamma1 + mu;   gam2_mu = gamma2 + mu;

if  (printyes)
    fprintf('\n *****************************************************');
    fprintf('******************************************');
    fprintf('\n ************** SC_PAMM for low-rank recovery problems ***********************');
    fprintf('\n ****************************************************');
    fprintf('*******************************************');
    fprintf('\n  iter      rankX       relerr      measure         obj          time ');
end

%% ***************** Initialization **************************

obj_list = zeros(maxiter,1);    

%% ************************* Main Loop *********************************

tstart = clock;

dsqrt = d.^(1/2);

for iter=1:maxiter
    
    %% ********************* Given B, to solve A **********************
       
    dmu_sqrt = (eta*d+gam1_mu).^(1/2);
    
    bk = dsqrt./dmu_sqrt;
        
    temp_Gk = (Zk*Q + gamma1*P).*bk;
    
    Gk_cnorm = dot(temp_Gk,temp_Gk).^(1/2);

    temp = Gk_cnorm./dmu_sqrt;

    gk = subp_Wsolver(temp, lambda, dmu_sqrt.^2, q);

    ind = gk>1.0e-8;
    
    rankU = sum(ind);
    
    if (rankU==0)
        
        disp('lambda is too large,please reset it again')
        
        return;
    end

    if rankU<length(gk)

        UD = temp_Gk(:,ind).*(gk(ind)./Gk_cnorm(ind).*dsqrt(ind));   % U*D

        X = UD*Q(:,ind)';   Xsnz = X(spidx);

        [P,D,L] = svd(UD,'econ');
    
        Q = Q(:,ind)*L;

    else

        UD = temp_Gk.*(gk./Gk_cnorm.*dsqrt);   % U*D

        X = UD*Q';   Xsnz = X(spidx);

        [P,D,L] = svd(UD,'econ');
    
        Q = Q*L;

    end
  
    d = diag(D)';    dsqrt = d.^(1/2);
    
    gradX = fun_grad(Xsnz);
    
    Zk = eta*X - gradX;

    %% ****************** Given A, to solve B *************************
    
    dmu_sqrt = (eta*d+gam2_mu).^(1/2);
    
    bk = dsqrt./dmu_sqrt;
            
    temp_Hk = (Zk'*P + gamma2*Q).*bk;
    
    Hk_cnorm = dot(temp_Hk,temp_Hk).^(1/2);

    temp = Hk_cnorm./dmu_sqrt;

    gk = subp_Wsolver(temp, lambda, dmu_sqrt.^2, q);

    ind = gk>1.0e-8;

    if (sum(ind)==0)
        
        disp('lambda is too large,please reset it again')
        
        return;
    end

    if sum(ind)<length(gk)

        VD = temp_Hk(:,ind).*(gk(ind)./Hk_cnorm(ind).*dsqrt(ind));  % V*D

        X = P(:,ind)*VD'; Xsnz = X(spidx);

        [Q,D,L] = svd(VD,'econ');

        P = P(:,ind)*L;

    else

        VD = temp_Hk.*(gk./Hk_cnorm.*dsqrt);  % V*D

        X = P*VD';

        Xsnz = X(spidx);

        [Q,D,L] = svd(VD,'econ');

        P = P*L;

    end
    
    d = diag(D)';     dsqrt = d.^(1/2);
    
    [gradX,loss] = fun_grad(Xsnz);
        
    Zk = eta*X - gradX;
    
    obj =  loss + mu*sum(d) + 2*lambda*fval_Lq(dsqrt,q);
    
    obj_list(iter) = obj;

    rankX = sum(ind);
    
    %% **************** check the stopping criterion ******************
    
    time = etime(clock,tstart);
    
    measure = norm(X-Xold,'fro')/max(1,norm(X,'fro'));
    
    if (printyes)&&(mod(iter,10)==0)

        relerr = norm(X-Mstar,'fro')/norm(Mstar,'fro');
        
        fprintf(' \n %2d          %2d        %3.2e      %3.2e      %3.5e     %3.2f \n',iter,rankX,relerr,measure,obj,time);
        
    end
  
    if (measure<tol) ||(iter>=10&&max(abs(obj-obj_list(iter-9:iter)))<=objtol*max(1,obj))              
        return;
    end
    
    gamma1 = max(lgamma,gratio*gamma1);
    
    gam1_mu = gamma1 + mu;
    
    gamma2 = max(lgamma,gratio*gamma2);
    
    gam2_mu = gamma2 + mu;

    Xold = X;
    
end

if (iter==maxiter)
    
    fprintf('\n maxiter');
    
end

end

