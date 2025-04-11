%% ************************************************************************
%% run random 1-bit matrix completion problems. 
%% ************************************************************************

clear;

addpath(genpath('solvers_Irate'));

addpath(genpath('Function'));

%% generate random a test problem

nr = 2000;  nc = 2000;

r = 10; 

type_phi = 1;   % Logistic model:1 /Gaussian noise:2 

q = 0;  

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

OPTIONS.Xmaxiter = 1000;

OPTIONS.UVmaxiter = 1000;

OPTIONS.Xtol = 5.0e-4;

OPTIONS.UVtol= 5.0e-4;

OPTIONS.objtol= 1.0e-6;

OPTIONS.printyes = 1;

cn =  [  10     30     80    100     150    200     300    400    500    800   1000  ];

lam = [3.75    4.03   3.85   3.93    3.74    3.86   4.08   4.01   3.85   3.85  3.85  ];

%% ******************** main loop  **********************************************

SR = 0.4;     

for i = 7:length(cn)
          
        randstate =  100*i
        randn('state',double(randstate));
        rand('state',double(randstate));
        
        fprintf('\n nr = %2.0d,  nc = %2.0d, rank = %2.0d\n,',nr,nc,r);
        
        num_sample = round(SR*nr*nc);     %% number of sampled entries
        
        %% *************** to generate the true matrix ******************
        
        X.U = rand(nr,r)-.5;        X.V = rand(nc,r)-.5;
        
        Mstar = X.U*X.V';           Mstar = Mstar/max(max(abs(Mstar(:))),1)*cn(i);  %30
        
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

        lambda = lam(i)*max(Mhat_cnorm)

        [iter,obj] = SC_PAMM_Irate(Mstar,Xstart,Zstart,Ustart,Vstart,dstart,nzidx,lambda,mu,pars,fun_grad,OPTIONS);

        Xmajor_times(i) = ttime0 + etime(clock,tstart);

        Xmajor_iter(i) = iter;

        Xmajor_obj(i) = obj;

      %% ********************* UV_Major ******************************

      tstart = clock;

      lambda =lam(i)*max(Mhat_cnorm)

      [iter,UVmajor_obj]= PALM_Log_Irate(Mstar,Ustart,Vstart,bb,bar,f,nzidx,OPTIONS,pars,lambda,mu);

      UVmajor_times(i) = ttime0 + etime(clock,tstart);

      UVmajor_iter(i) = iter;

      UVmajor_obj(i) = obj;

end

save('X_result_Lp','Xmajor_iter','Xmajor_obj','Xmajor_times')

save('UV_result_Lp','UVmajor_iter','UVmajor_obj','UVmajor_times')
%% *************************************************************************

subplot(1,2,1);
h1=plot(cn,Xmajor_iter,'r*-',cn,UVmajor_iter,'b+-');
xlabel('$c_{M^*}$','FontSize',14,'FontWeight','bold');   ylabel('iteration','FontSize',14,'FontWeight','bold');
legend('Algorithm 1','PALM','FontSize',10.5);
grid on;

subplot(1,2,2);
h2=plot(cn,Xmajor_times,'r*-',cn,UVmajor_times,'b+-');
xlabel('$c_{M^*}$','FontSize',14,'FontWeight','bold');   ylabel('Time(s)','FontSize',14,'FontWeight','bold');
legend('Algorithm 1','PALM','FontSize',10.5);
grid on;

