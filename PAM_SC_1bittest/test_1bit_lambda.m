%% ************************************************************************
%% run random 1-bit matrix completion problems. 
%% ************************************************************************
% type_phi = 1;   q = 0;
% type_phi = 2;   q = 1/2; 

clear;
addpath(genpath('solvers'));
addpath(genpath('Function'));
%
%% generate random a test problem

nr = 2000;  nc = 2000;

r = 10; 

q = 0;    %1/2

type_phi = 1;   % Logistic model:1  /Laplacian noise:2   

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

OPTIONS.printyes = 300;

OPTIONS_Hybrid.maxiter = 300;
        
OPTIONS_Hybrid.printyes = 1;

OPTIONS_Hybrid.tol = 5.0e-2;

%% ******************** main loop  **********************************************

SR = 0.4;   

Xnu_type1 = [  3   3.2   3.4   3.5   3.55   3.6   3.65   3.68  3.7  3.72  3.75  3.78  3.8  3.82  3.85  4 ];   

UVnu_type1 = [ 3   3.2   3.4   3.5   3.55   3.6   3.65   3.68  3.7  3.72  3.75  3.78  3.8  3.82  3.85  4 ];


Xnu_type2 = [ 1.35  1.36  1.38  1.4   1.42  1.44  1.45  1.455   1.46   1.465  1.47   1.475  1.485   1.49  1.5  1.51  ]  ;

UVnu_type2 =[ 1.35  1.36  1.38  1.4   1.42  1.44  1.45  1.455   1.46   1.465  1.47   1.475  1.485   1.49  1.5  1.51 ];

Xnu = Xnu_type1;

UVnu = UVnu_type1;

ntest = 5;

for i = 1:length(Xnu)
    
    i
    
    for t=1:ntest
        
        t
          
        randstate =  1000*i*t
        randn('state',double(randstate));
        rand('state',double(randstate));
        
        fprintf('\n nr = %2.0d,  nc = %2.0d, rank = %2.0d\n,',nr,nc,r);
        
        num_sample = round(SR*nr*nc);     %% number of sampled entries
        
        %% *************** to generate the true matrix ******************la
        
        X.U = rand(nr,r)-.5;        X.V = rand(nc,r)-.5;
        
        Mstar = X.U*X.V';           Mstar = Mstar/max(max(abs(Mstar(:))),1)*30; %*30
        
        normM = norm(Mstar,'fro');
        
       %% Obtain signs of noisy measurements
       
        B = sign(f(Mstar)-rand(nr,nc));
        
       %%  ***********  uniform sampling  ***************************
       
        idx = randperm(nr*nc);
       
        nzidx = idx(1:num_sample)';
       
        zidx = idx(num_sample+1:end)';
       
        bb =  B(nzidx);       bar = (bb-1)/2;
        
        Mhat = zeros(nr,nc);  Mhat(nzidx) = bb;  
        
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
                 LipX = 1; %0.8*max(fprime(Xsnz));  
                 
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

        lambda = Xnu(i)*max(Mhat_cnorm);   %*

        ttime1 =  etime(clock,tstart);

        tstart = clock;

        [rankX,Xopt,Xobj] = SC_PAMM(Mstar,Xstart,Zstart,Ustart,Vstart,dstart,nzidx,lambda,mu,pars,fun_grad,OPTIONS);

        ttime = ttime0 + ttime1 + etime(clock,tstart);

        relerr = norm(Xopt-Mstar,'fro')/normM;

        Xmajor_obj(i,t) = Xobj;

        Xmajor_RMSE(i,t) = relerr;

        Xmajor_RankX(i,t) = rankX

        Xmajor_time(i,t) = ttime;

      %% ********************** Hybrid_solver *************************

       % tstart = clock;
       % 
       %  OPTIONS_Hybrid.Lip_const = LipX;
       % 
       %  [rankX,Uinit,Vinit] = Hybrid_initial(Mstar,Zstart,Ustart,Vstart,dstart,nzidx,lambda,mu,pars,fun_grad,OPTIONS_Hybrid);
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
       %  Brid_RMSE(i,t) = relerr;
       % 
       %  Brid_RankX(i,t) = rankX;
       % 
       %  Brid_time(i,t) = time;

      %% ********************* UV_Major without linear search ******************************
      
        tstart = clock;
        
        mu = 1.0e-8;
        
        lambda = UVnu(i)*max(Mhat_cnorm);

        switch  type_phi
    
            case 1
               %%  Logistic model
                 [UVopt,rankXopt,UVobj]= PALM_Log(Mstar,Ustart,Vstart,bb,bar,f,nzidx,OPTIONS,pars,lambda,mu);  
                 
             case 2 

                 [UVopt,rankXopt,UVobj]= PALM(Mstar,Ustart,Vstart,bb,bar,f,fprime,nzidx,OPTIONS,pars,lambda,mu);
        end
       
        ttime = ttime0 + etime(clock,tstart);
        
        relerr = norm(UVopt-Mstar,'fro')/normM;
        
        UVmajor_obj(i,t) = UVobj;
        
        UVmajor_RMSE(i,t) = relerr;
        
        UVmajor_RankX(i,t) = rankXopt
        
        UVmajor_time(i,t) = ttime;
   
    end   
    
       
     Xmajor_aveRMSE(i) =  sum(Xmajor_RMSE(i,:))/ntest

     Xmajor_aveRankX(i) =  sum(Xmajor_RankX(i,:))/ntest

     Xmajor_avetime(i) =  sum(Xmajor_time(i,:))/ntest;

     Xmajor_aveobj(i) =  sum(Xmajor_obj(i,:))/ntest;


     % Brid_aveRMSE(i) =  sum(Brid_RMSE(i,:))/ntest
     % 
     % Brid_aveRankX(i) =  sum(Brid_RankX(i,:))/ntest
     % 
     % Brid_avetime(i) =  sum(Brid_time(i,:))/ntest;
     % 
     % Brid_aveobj(i) =  sum(Brid_obj(i,:))/ntest;
     
    
     UVmajor_aveRMSE(i) =  sum(UVmajor_RMSE(i,:))/ntest

     UVmajor_aveRankX(i) =  sum(UVmajor_RankX(i,:))/ntest

     UVmajor_avetime(i) =  sum(UVmajor_time(i,:))/ntest;

     UVmajor_aveobj(i) =  sum(UVmajor_obj(i,:))/ntest;
    
end

% 
save('X_result','Xnu','Xmajor_aveRMSE','Xmajor_aveRankX','Xmajor_avetime','Xmajor_aveobj')

 save('Brid_result','Xnu','Brid_aveRMSE','Brid_aveRankX','Brid_avetime','Brid_aveobj')

 save('UV_result','UVnu','UVmajor_aveRMSE','UVmajor_aveRankX','UVmajor_avetime','UVmajor_aveobj')
%% *************************************************************************

set(0,'defaulttextinterpreter','latex'); 


%% ***************************************************************

nu = Xnu;
subplot(1,3,1);
h1=plot(nu,Xmajor_aveRMSE,'rs-',nu,UVmajor_aveRMSE,'b+-',nu,Brid_aveRMSE,'go-');
xlabel('$c_{\lambda}$','FontSize',12,'FontWeight','bold');   ylabel('RE','FontSize',12,'FontWeight','bold');
legend('Algorithm 1','PALM','Hybrid AMM','FontSize',10.5);
set(h1,'Linewidth',1.5);
grid on;


subplot(1,3,2);
h2=plot(nu,Xmajor_aveRankX,'rs-',nu,UVmajor_aveRankX,'b+-',nu,Brid_aveRankX,'go-');
xlabel('$c_{\lambda}$','FontSize',12,'FontWeight','bold');   ylabel('Rank','FontSize',12,'FontWeight','bold');
legend('Algorithm 1','PALM','Hybrid AMM','FontSize',10.5);
set(h2,'Linewidth',1.5);
grid on;


subplot(1,3,3);
h4=plot(nu,Xmajor_avetime,'rs-',nu,UVmajor_avetime,'b+-',nu,Brid_avetime,'go-');
xlabel('$c_{\lambda}$','FontSize',12,'FontWeight','bold');   ylabel('Time(s)','FontSize',12,'FontWeight','bold');
legend('Algorithm 1','PALM','Hybrid AMM','FontSize',10.5);
set(h4,'Linewidth',1.5);
grid on;
