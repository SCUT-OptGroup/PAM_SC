%% ********************************************************************
%  filename: PALM_Log_constlip for Logistic noise
%
%% **********************************************************************
%% Alternating majorization-minimization method for solving
%
%  min{ -sum P_Omega[log(Y.*f(UV')-(Y-1)/2)]+ 0.5*mu*(||U||_F^2+||V||_F^2)}+lambda(||U||_{2,0}+||V||_{2,0}) (*)
%  
%  where f(X) is Logistic noise
%  
% in the order: U--->V---->U
%
%% **********************************************************************
%%  2024-03-20
%% *************************************************************************

function [Xnew,rankX,obj] = PALM_Log(Mstar,U,V,y,ybar,f,spidx,OPTIONS,pars,lambda,mu)

if isfield(OPTIONS,'UVtol');        UVtol     = OPTIONS.UVtol;         end
if isfield(OPTIONS,'objtol');       objtol    = OPTIONS.objtol;        end
if isfield(OPTIONS,'printyes');     printyes  = OPTIONS.printyes;      end
if isfield(OPTIONS,'UVmaxiter');    maxiter   = OPTIONS.UVmaxiter;     end
if isfield(OPTIONS,'Lip_const');    LipX      = OPTIONS.Lip_const;     end

nr= pars.nr;  nc = pars.nc;   r = pars.k;    q = pars.q;

if  (printyes)
    fprintf('\n *****************************************************');
    fprintf('******************************************');
    fprintf('\n ************** PALM without Lip-constant searching  ********************');
    fprintf('\n ****************************************************');
    fprintf('*******************************************');
    fprintf('\n  iter    rankX      LipU      LipV      optmeasure        fval        time      relerr');
end

%% ************************* Main Loop *********************************

tstart = clock;

obj_list = zeros(maxiter,1);    

Uold = U;   Vold = V;

Xold = Uold*Vold';

[~,gradU] = funU_log(V,y,ybar,spidx,f,nr,nc,Xold);

LipU = LipX*norm(V,2)^2;  tauU = LipU + mu;

for iter = 1:maxiter  
       
  %% ****************** to compute Unew *************************
    
    Utemp = (1/tauU)*(LipU*U-gradU);
    
    Utemp_cnorm = dot(Utemp,Utemp).^(1/2);

    ind = Utemp_cnorm>0;
        
    Unew = zeros(nr,r);

    gk = subp_solver(Utemp_cnorm(:,ind), lambda, tauU, q);

    Unew(:,ind) = Utemp(:,ind).*gk./Utemp_cnorm(:,ind);
    
    rankU = length(gk(gk>0));
    
    if rankU==0

        disp('lambda is too large,please reset it again')
         
        return;
        
    end
    
    UnewVt = Unew*V';
    
    LipV = LipX*norm(Unew,2)^2;   tauV = LipV + mu;

  %% ****************** to compute Vnew **************************

    [~,gradV] = funV_log(Unew,y,ybar,spidx,f,nr,nc,UnewVt);
        
    Vtemp = (1/tauV)*(LipV*V-gradV);
    
    Vtemp_cnorm = dot(Vtemp,Vtemp).^(1/2);

    ind = Vtemp_cnorm>0;
        
    Vnew = zeros(nc,r);

    gk = subp_solver(Vtemp_cnorm(:,ind), lambda, tauV, q);

    Vnew(:,ind) = Vtemp(:,ind).*gk./Vtemp_cnorm(:,ind);

    rankV = length(gk(gk>0));
    
    Xnew = Unew*Vnew';
    
    rankX = min(rankU,rankV);
        
%% ******************** Optimality checking ************************
    
    [Loss,gradU] = funUV_log(Unew,Vnew,y,ybar,spidx,f,nr,nc,Xnew);

    Unorm = sum(Unew.*Unew).^(1/2); 

    Vnorm = sum(Vnew.*Vnew).^(1/2);
    
    obj = Loss + 0.5*mu*(norm(Unew,'fro')^2 + norm(Vnew,'fro')^2) + lambda*(fval_Lq(Unorm,q)+fval_Lq(Vnorm,q));
           
    obj_list(iter) = obj;
    
    ttime = etime(clock,tstart);

    measure = norm(Xnew-Xold,'fro')/max(1,norm(Xnew,'fro'));
   
    if (printyes && (mod(iter,10)==0))
        
        relerr = norm(Xnew-Mstar,'fro')/norm(Mstar,'fro');
        
        fprintf('\n  %2d      %2d     %3.2e    %3.2e     %3.2e     %3.4e      %3.2f    %3.2e',iter,rankX,LipU,LipV,measure,obj,ttime,relerr);

    end
          
    if (measure<UVtol) ||(iter>=10&&max(abs(obj-obj_list(iter-9:iter)))<=objtol*max(1,obj))
            
            fprintf('\n  %2d       %2d      %3.2e    %3.2e     %3.2e      %3.4e     %3.2f',iter,rankX,LipU,LipV,measure,obj,ttime);
            
            return;
    end
    
   %% *******  estimating the Lip-constant of grad_U F(U,V) ************   
        
    LipU = LipX*norm(Vnew,2)^2;   tauU = LipU + mu;
    
    Xold = Xnew;    U = Unew;    V = Vnew;  
     
end

end

