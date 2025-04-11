%% ********************************************************************
%  filename: PALMbb_log for Logistic noise
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

function [Xnew,rankX,obj] = PALM_Log_ls(Mstar,U,V,y,ybar,f,spidx,OPTIONS,pars,lambda,mu)

if isfield(OPTIONS,'UVtol');        UVtol     = OPTIONS.UVtol;         end
if isfield(OPTIONS,'objtol');       objtol    = OPTIONS.objtol;        end
if isfield(OPTIONS,'printyes');     printyes  = OPTIONS.printyes;      end
if isfield(OPTIONS,'UVmaxiter');    maxiter   = OPTIONS.UVmaxiter;     end

nr= pars.nr;  nc = pars.nc;   r = pars.k;    q = pars.q;

LipU_org = 1;        LipV_org = 1;

beta1 = 1e-10;       beta2 = 1.0e10;

ratio = 5;

if  (printyes)
    fprintf('\n *****************************************************');
    fprintf('******************************************');
    fprintf('\n ************** PALM_ls   ********************');
    fprintf('\n ****************************************************');
    fprintf('*******************************************');
    fprintf('\n  iter    rankX     nU      nV        LipU         LipV      optmeasure        fval        time      relerr');
end

%% ************************* Main Loop *********************************

tstart = clock;

obj_list = zeros(maxiter,1);    

Uold = U;   Vold = V;

Xold = Uold*Vold';

[Loss,gradU] = funU_log(V,y,ybar,spidx,f,nr,nc,Xold);

for iter = 1:maxiter  
       
  %% ****************** to compute Unew *************************
    
    LipU = LipU_org;  tauU = LipU + mu; 
 
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
     
 %% ****************** to search the Lip-constant *******************
    
    UnewVt = Unew*V';
    
    Loss_Unew = funU_log(V,y,ybar,spidx,f,nr,nc,UnewVt);
    
    diffU = Unew - U;
    
    nU = 0;
    
    Lip_check = Loss + sum(dot(gradU,diffU))+ 0.5*LipU*norm(diffU,'fro')^2;
    
    while (Loss_Unew>Lip_check)
        
        nU = nU + 1;
        
        LipU = (ratio)^nU*LipU_org;
        
        tauU = LipU + mu;
        
        Utemp = (1/tauU)*(LipU*U-gradU);
        
        Utemp_cnorm = dot(Utemp,Utemp).^(1/2);

        ind = Utemp_cnorm>0;
        
        Unew = zeros(nr,r);

        gk = subp_solver(Utemp_cnorm(:,ind), lambda, tauU, q);

        Unew(:,ind) = Utemp(:,ind).*gk./Utemp_cnorm(:,ind);

        rankU = length(gk(gk>0));
        
        UnewVt = Unew*V';
        
        Loss_Unew = funU_log(V,y,ybar,spidx,f,nr,nc,UnewVt);
        
        diffU = Unew - U;
        
        Lip_check = Loss + sum(dot(gradU,diffU))+ 0.5*LipU*norm(diffU,'fro')^2;
        
    end
    
  %% ****************** to compute Vnew **************************
    if iter==1       
       
        LipV = LipV_org;    tauV = LipV + mu;
        
        [Loss,gradV] = funV_log(Unew,y,ybar,spidx,f,nr,nc,UnewVt);
        
    else
        [~,gradVold] = funV_log(Unew,y,ybar,spidx,f,nr,nc,Unew*Vold');
       
        [Loss,gradV] = funV_log(Unew,y,ybar,spidx,f,nr,nc,UnewVt);
       
        LipV_org = norm(gradV-gradVold,'fro')/norm(V-Vold,'fro');
    
        LipV_org = min(max(LipV_org,beta1),beta2);
        
        LipV = LipV_org;    tauV = LipV + mu;
    end    
        
    Vtemp = (1/tauV)*(LipV*V-gradV);
    
    Vtemp_cnorm = dot(Vtemp,Vtemp).^(1/2);

    ind = Vtemp_cnorm>0;
        
    Vnew = zeros(nc,r);

    gk = subp_solver(Vtemp_cnorm(:,ind), lambda, tauV, q);

    Vnew(:,ind) = Vtemp(:,ind).*gk./Vtemp_cnorm(:,ind);

    rankV = length(gk(gk>0));
    
 %% ****************** to search the Lip-constant  *******************
    
    Xnew = Unew*Vnew';
    
    Loss_Vnew =  funV_log(Unew,y,ybar,spidx,f,nr,nc,Xnew);
    
    diffV = Vnew - V;
    
    Lip_check = Loss + sum(dot(gradV,diffV))+ 0.5*LipV*norm(diffV,'fro')^2;
    
    nV = 0;  
    
    while (Loss_Vnew>Lip_check)
        
        nV = nV + 1;
        
        LipV = (ratio)^nV*LipV_org;
        
        tauV = LipV + mu;
        
        Vtemp = (1/tauV)*(LipV*V-gradV);

        Vtemp_cnorm = dot(Vtemp,Vtemp).^(1/2);

        ind = Vtemp_cnorm>0;

        Vnew = zeros(nc,r);

        gk = subp_solver(Vtemp_cnorm(:,ind), lambda, tauV, q);

        Vnew(:,ind) = Vtemp(:,ind).*gk./Vtemp_cnorm(:,ind);

        rankV = length(gk(gk>0));

        Xnew = Unew*Vnew';
        
        Loss_Vnew =  funV_log(Unew,y,ybar,spidx,f,nr,nc,Xnew);
        
        diffV = Vnew - V;
        
        Lip_check = Loss + sum(dot(gradV,diffV)) + 0.5*LipV*norm(diffV,'fro')^2;
        
    end
    
    rankX = min(rankU,rankV);
        
%% ******************** Optimality checking ************************
    
    [Loss,gradUnew] = funUV_log(Unew,Vnew,y,ybar,spidx,f,nr,nc,Xnew);

    Unorm = sum(Unew.*Unew).^(1/2); 

    Vnorm = sum(Vnew.*Vnew).^(1/2);
    
    obj = Loss + 0.5*mu*(norm(Unew,'fro')^2 + norm(Vnew,'fro')^2) + lambda*(fval_Lq(Unorm,q)+fval_Lq(Vnorm,q));
           
    obj_list(iter) = obj;
    
    ttime = etime(clock,tstart);

    measure = norm(Xnew-Xold,'fro')/max(1,norm(Xnew,'fro'));
   
    if (printyes && (mod(iter,10)==0))
        
        relerr = norm(Xnew-Mstar,'fro')/norm(Mstar,'fro');
        
        fprintf('\n  %2d        %2d      %2d       %2d     %3.2e    %3.2e    %3.2e     %3.4e      %3.2f    %3.2e',iter,rankX,nU,nV,LipU,LipV,measure,obj,ttime,relerr);

    end
          
    if (measure<UVtol) ||(iter>=10&&max(abs(obj-obj_list(iter-9:iter)))<=objtol*max(1,obj))
            
            fprintf('\n  %2d      %2d      %2d       %2d      %3.2e    %3.2e    %3.2e      %3.4e     %3.2f',iter,rankX,nU,nV,LipU,LipV,measure,obj,ttime);
            
            return;
    end
    
   %% ******* BB step for estimating the Lip-constant of grad_U F(U,V) ************   
       
   [~,gradU] = funU_log(Vnew,y,ybar,spidx,f,nr,nc,U*Vnew');
        
    LipU_org = norm(gradUnew-gradU,'fro')/norm(Unew-U,'fro');  %10
    
    LipU_org = min(max(LipU_org,beta1),beta2);
    
 %% ******* BB step for estimating strat Lip-constant for grad_V F(U,V) ************
   
    Xold = Xnew; 
    
    Vold=V;    V = Vnew;  
    
    U = Unew;  gradU = gradUnew;  
end

end

