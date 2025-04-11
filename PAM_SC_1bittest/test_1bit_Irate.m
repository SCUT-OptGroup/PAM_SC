
%% ************************************************************************
%% run random 1-bit matrix completion problems. 
%% ************************************************************************
%% type_phi = 1  %0,   1/2,  1,  % lambda =  3.75,  1.08,   0.50
%% type_phi = 2  %0,   1/2,  1,  % lambda =  7.7,   1.5,    0.50


clear;

addpath(genpath('solvers_Irate'));

addpath(genpath('Function'));

%% generate random a test problem

nr = 2000;  nc = 2000;

r = 10; 

type_phi = 2;   % Logistic model:1 /Gaussian noise:2 

q = 1;  %0,   1/2,  1,  % lambda =  3.75,  1.08,   0.50

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

OPTIONS.Xmaxiter = 500;

OPTIONS.UVmaxiter = 500;

OPTIONS.Xtol = 1.0e-6;

OPTIONS.UVtol= 1.0e-6;

OPTIONS.objtol= 1.0e-6;

OPTIONS.printyes = 1;

%% ******************** main loop  **********************************************

SR = 0.4;     

for i = 1
          
        randstate =  100*i
        randn('state',double(randstate));
        rand('state',double(randstate));
        
        fprintf('\n nr = %2.0d,  nc = %2.0d, rank = %2.0d\n,',nr,nc,r);
        
        num_sample = round(SR*nr*nc);     %% number of sampled entries
        
        %% *************** to generate the true matrix ******************
        
        X.U = rand(nr,r)-.5;        X.V = rand(nc,r)-.5;
        
        Mstar = X.U*X.V';           Mstar = Mstar/max(max(abs(Mstar(:))),1)*30;  %30
        
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
        
        k = 10*r;      pars.q = q;    mu = 1.0e-8;
        
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
                 LipX = 1; %1   %0.8*max(fprime(Xsnz));   % 1

            case 2
               %%  Laplacian model    
                 LipX = 2/a^2;
                 
            case 3
               %% Gaussian model
                 fbb = min(abs (f(bb(:)) + bar));     LipX = 2*(1/(2*sigma^2*pi*exp(1)*fbb) + 1/(2*pi*fbb^2)) ;
                   
        end
        
        OPTIONS.Lip_const= LipX;    

        gradX = fun_grad(Xsnz);       

        Zstart = LipX*Xstart- gradX;   % X-gradfXk

        lambda = 0.5*max(Mhat_cnorm);     

       [Xmajor_iter1,~,Xmajor_times1,Xmajor_obj1,Xmajor_diffX1] = SC_PAMM_Irate(Mstar,Xstart,Zstart,Ustart,Vstart,dstart,nzidx,lambda,mu,pars,fun_grad,OPTIONS);
                                                              

      %%  UV_Major with Lip-constant  searching

       lambda = 0.5*max(Mhat_cnorm); 

       switch  type_phi

           case 1
               %%  Logistic model

               [UVmajorLS_iter20,~,UVmajorLS_times20,UVmajorLS_obj20,UVmajorLS_diffX20]= PALM_Log_Irate(Mstar,Ustart,Vstart,bb,bar,f,nzidx,OPTIONS,pars,lambda,mu);

           case 2
               [UVmajorLS_iter1,~,UVmajorLS_times1,UVmajorLS_obj1,UVmajorLS_diffX1]= PALM_Irate(Mstar,Ustart,Vstart,bb,bar,f,fprime,nzidx,OPTIONS,pars,lambda,mu);

       end

      %% ********************* UV_Major ******************************
   
        lambda = 0.5*max(Mhat_cnorm);   

       [UVmajor_iter1,~,UVmajor_times1,UVmajor_obj1,UVmajor_diffX1]= PALM_Log_lsIrate(Mstar,Ustart,Vstart,bb,bar,f,nzidx,OPTIONS,pars,lambda,mu);

end


save('X_result_Lp1','Xmajor_iter1','Xmajor_obj1','Xmajor_diffX1','Xmajor_times1')

save('UVLS_result_Lp1','UVmajorLS_iter1','UVmajorLS_obj1','UVmajorLS_diffX1','UVmajorLS_times1')

save('UV_result_Lp1','UVmajor_iter1','UVmajor_obj1','UVmajor_diffX1','UVmajor_times1')
%% *************************************************************************
