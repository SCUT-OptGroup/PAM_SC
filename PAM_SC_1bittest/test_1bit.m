%% ************************************************************************
%% run random 1-bit matrix completion problems. 
%% ************************************************************************

clear;
addpath(genpath('solvers'));
addpath(genpath('Function'));
%
%% generate random a test problem

ntest = 5;

nr = 1000;  nc = 1000;

r = 10; 

q = 1/2;  

type_phi = 2;   % Logistic model:1  /Laplacian noise:2   

switch type_phi
    
    case 1
        
        %% Define observation model Logistic model
        f       = @(x) (1 ./ (1 + exp(-x)));
        fprime  = @(x) (exp(x) ./ (1 + exp(x)).^2);
        
    case 2

        %% Define observation model (Laplacian noise)
        a = 2;
        f      = @(x) Laplacdf(x,a);
        fprime = @(x) Laplapdf(x,a);
    
end
%% 

OPTIONS.Xmaxiter = 300;

OPTIONS.Xtol = 3.0e-3;

OPTIONS.UVmaxiter = 300;

OPTIONS.UVtol= 3.0e-3;

OPTIONS.objtol= 1.0e-3;

OPTIONS.printyes = 1;

OPTIONS_Hybrid.maxiter = 300;
        
OPTIONS_Hybrid.printyes = 1;

OPTIONS_Hybrid.tol = 5.0e-2;

%% ******************** main loop  **********************************************

nSR = [0.3   0.35   0.4 ];     ns = length(nSR);

Xnu  =  [1.10   1.18  1.22] 

UVnu =  [1.11   1.19  1.23]

for i = 1:3
    
    i

    SR = nSR(i)
    
    for t=1:ntest
        
        t
          
        randstate =  1000*i*t
        randn('state',double(randstate));
        rand('state',double(randstate));
        
        fprintf('\n nr = %2.0d,  nc = %2.0d, rank = %2.0d\n,',nr,nc,r);
        
        num_sample = round(SR*nr*nc);     %% number of sampled entries
        
        %% *************** to generate the true matrix ******************la
        
        X.U = rand(nr,r)-.5;        X.V = rand(nc,r)-.5;
        
        Mstar = X.U*X.V';           Mstar = Mstar/max(max(abs(Mstar(:))),1)*30;
        
        normM = norm(Mstar,'fro');
        
       %% Obtain signs of noisy measurements
       
        B = sign(f(Mstar)-rand(nr,nc));
        
       %%  ***********  uniform sampling  ***************************
       
        idx = randperm(nr*nc);
       
        nzidx = idx(1:num_sample)';
       
        zidx = idx(num_sample+1:end)';
       
        bb =  B(nzidx);        bar = (bb-1)/2;
        
        Mhat = zeros(nr,nc);   Mhat(nzidx) = bb;  
        
        Mhat_cnorm = sum(Mhat.*Mhat).^(1/2);
        
        fun_grad = @(x)funX_grad(x,bb,bar,nzidx,f,fprime,nr,nc);
        
    %% *************** Initialization part *********************
        
        k = 10*r;      pars.q = q;
        
        pars.k = k;   pars.nc = nc;  pars.nr = nr; 
        
        pars.normM = normM;   pars.normb = norm(bb);
        
       %% ************ generate Initial   point  ***************************
        
        tstart = clock;
        
        Pstart = randn(nr,k);     Qstart = randn(nc,k); 
        
        Ustart = orth(Pstart);    Vstart = orth(Qstart);
        
        ttime0 = etime(clock,tstart);
        
     %% ********************* X_Major ******************************
        tstart = clock;

        Xstart = Ustart*Vstart';  dstart = ones(1,k); %  Xstart =Ustart*Vstart';

        Xsnz = Xstart(nzidx);  

   %% ****** estimate lip constant of grad f(X)  ****************************

        switch  type_phi

            case 1
               %%  Logistic model
                 LipX = 1; % 0.8*max(fprime(Xsnz));  

            case 2
               %%  Laplacian model
                 LipX = 2/a^2; 

            case 3
               %% Gaussian model
                fbb = min(abs (f(bb(:)) + bar));     LipX = (1/(2*sigma^2*pi*exp(1)*fbb) + 1/(2*pi*fbb^2)) ;
        end

         OPTIONS.Lip_const= LipX;    

        gradX = fun_grad(Xsnz);       

        Zstart = LipX*Xstart- gradX;   % X-gradfXk

        mu = 1.0e-8;

        ttime1 =  etime(clock,tstart);

        tstart = clock;

        lambda = Xnu(i)*max(Mhat_cnorm)   %*  

        [rankX,Xopt,Xobj] = SC_PAMM(Mstar,Xstart,Zstart,Ustart,Vstart,dstart,nzidx,lambda,mu,pars,fun_grad,OPTIONS);

        ttime = ttime0 + ttime1 + etime(clock,tstart);

        relerr = norm(Xopt-Mstar,'fro')/normM

        Xmajor_obj(i,t) = Xobj;

        Xmajor_RMSE(i,t) = relerr

        Xmajor_time(i,t) = ttime

        Xmajor_RankX(i,t) = rankX

      %% ********************** Hybrid_solver *************************

       % tstart = clock;
       % 
       % OPTIONS_Hybrid.Lip_const = LipX;
       % 
       % [rankX,Uinit,Vinit] = Hybrid_initial(Mstar,Zstart,Ustart,Vstart,dstart,nzidx,lambda,mu,pars,fun_grad,OPTIONS_Hybrid);
       % 
       %  switch  type_phi
       % 
       %      case 1
       %         %%  Logistic model
       %           [Xopt,Xobj] = Hybrid_smooth_log(Mstar,Uinit,Vinit,bb,bar,f,nzidx,OPTIONS,pars,mu);  
       % 
       %       case 2 
       % 
       %           [Xopt,Xobj]= Hybrid_smooth(Mstar,Uinit,Vinit,bb,bar,f,fprime,nzidx,OPTIONS,pars,mu);
       % 
       %  end
       % 
       %  time = ttime0 + ttime1+  etime(clock,tstart);
       % 
       %  relerr = norm(Xopt- Mstar,'fro')/normM;
       % 
       %  Brid_obj(i,t) = Xobj;
       % 
       %  Brid_RMSE(i,t) = relerr
       % 
       %  Brid_RankX(i,t) = rankX
       % 
       %  Brid_time(i,t) = time

     %% ********************* UV_Major ******************************

        tstart = clock;

        mu = 1.0e-8;

        lambda = UVnu(i)*max(Mhat_cnorm) 

        switch  type_phi

            case 1
               %  Logistic model

                 [UVopt,rankXopt,UVobj]= PALM_Log(Mstar,Ustart,Vstart,bb,bar,f,nzidx,OPTIONS,pars,lambda,mu);  

             case 2 

                  [UVopt,rankXopt,UVobj]= PALM(Mstar,Ustart,Vstart,bb,bar,f,fprime,nzidx,OPTIONS,pars,lambda,mu);
        end

        ttime = ttime0 + etime(clock,tstart);

        relerr = norm(UVopt-Mstar,'fro')/normM

        UVmajor_obj(i,t) = UVobj;

        UVmajor_RMSE(i,t) = relerr

        UVmajor_time(i,t) = ttime

        UVmajor_RankX(i,t) = rankXopt

    end
    
end


Xmajor_aveRMSE =  mean(Xmajor_RMSE,2)'

Xmajor_aveRankX =  mean(Xmajor_RankX,2)'

Xmajor_avetime =  mean(Xmajor_time,2)'

Xmajor_aveobj =  mean(Xmajor_obj,2)';

% 
% Brid_aveRMSE =  mean(Brid_RMSE,2)'
% 
% Brid_aveRankX =  mean(Brid_RankX,2)'
% 
% Brid_avetime =  mean(Brid_time,2)'
% 
% Brid_aveobj =  mean(Brid_obj,2)';


UVmajor_aveRMSE =  mean(UVmajor_RMSE,2)'

UVmajor_aveRankX =  mean(UVmajor_RankX,2)'

UVmajor_avetime =  mean(UVmajor_time,2)'

UVmajor_aveobj =  mean(UVmajor_obj,2)';

%% *************************************************************************
