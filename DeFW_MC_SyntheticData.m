clear all; close all; clc;

addpath('./Tools_Data');

% Dataset for Figure 1,2
load MC_SynData_Small 

% Dataset for Figure 3
% load MC_SynData_Small_R10

% Dataset for Figure 4
% load MC_SynData_Large 

idx_TEST = setdiff( 1:m1*m2, idx_TRAIN);
prm_VEC = vec(prm_bar);

% Swap in the noiseless data, comment this for Fig. 2
YTrain_N = YTrain_N_noiseless;
YTrain = YTrain_noiseless;

% construct the decen. mixing matrix (make it static first)
% pc = 0.1;
% W = (rand(N)<pc); W = W + eye(N); W = W + W'; W = (W>0);
load graph_N50 % load a pre-generated network
deg_vec = W*ones(N,1); T1 = repmat(1./deg_vec,1,N); T2 = min(T1,T1').*W;
WD = diag( ones(N,1) - T2*ones(N,1) ) + T2;

sigma = 5;
rho = 1.2*sum( svd(prm_bar) );

% define the gradient
grad_sq = @(x,y) (x-y);
f_sq = @(x,y) (x-y).^2;
grad_gau = @(x,y) (x-y).*exp(-(x-y).^2 / (2*sigma)) / sigma;
f_gau = @(x,y) 1 - exp( -(x-y).^2 / (2*sigma));

% Create the initial matrices for each agent [need only the stuffs from FULL]
val_X = zeros(TRAIN+TEST,N); val_Xbar = val_X; % for Decen+Gau
val_XCen = zeros(TRAIN+TEST,1); % for Cen+Gau
val_Xsq = zeros(TRAIN+TEST,N); val_Xbarsq = val_Xsq; % for Decen+Sq
val_XCensq = zeros(TRAIN+TEST,1);

val_Xdpg = zeros(TRAIN+TEST,N); val_Xbardpg = zeros(TRAIN+TEST,N);

no_iter = 10e3;
mse_train = zeros(no_iter,1); mse_test = zeros(no_iter,1); duality = zeros(no_iter,1);
mse_train_Cen = zeros(no_iter,1); mse_test_Cen = zeros(no_iter,1);
mse_trainsq = zeros(no_iter,1); mse_testsq = zeros(no_iter,1);
mse_train_Censq = zeros(no_iter,1); mse_test_Censq = zeros(no_iter,1);
mse_train_dpg = zeros(no_iter,1); mse_test_dpg = zeros(no_iter,1);

consense_err_dpg = zeros(no_iter,1);
consense_err_Gau = zeros(no_iter,1); consense_err_Sq = zeros(no_iter,1);
consense_err_grdGau = zeros(no_iter,1); consense_err_grdSq = zeros(no_iter,1);

obj_deGau = zeros(no_iter,1); obj_ceGau = zeros(no_iter,1);
obj_deSq = zeros(no_iter,1); obj_ceSq = zeros(no_iter,1);
obj_dpg = zeros(no_iter,1);

time_deGau = zeros(no_iter,1);
time_deSq = zeros(no_iter,1);
time_dpg = zeros(no_iter,1);

duality_cen = zeros(no_iter,1);

% set options
OPT.tol = 1e-6;

tic;
for iter = 1 : no_iter
    
    if mod(iter,10) == 2
        fprintf('Iter: %i, Gau (Decen) loss: MSE Train: %f, MSE Test: %f, Con: %f, Con(Gr): %f, Duality: %f \n', ...
            iter-1,mse_train(iter-1),mse_test(iter-1),consense_err_Gau(iter-1),consense_err_grdGau(iter-1), duality(iter-1));
        fprintf('          Objective value: %f, Time: %f \n', obj_deGau(iter-1), time_deGau(iter-1) );
        fprintf('Iter: %i, Gau (Cen) loss: MSE Train: %f, MSE Test: %f, Obj: %f \n', ...
            iter-1,mse_train_Cen(iter-1),mse_test_Cen(iter-1), obj_ceGau(iter-1) );
        fprintf('Iter: %i, Sq (Decen) loss: MSE Train: %f, MSE Test: %f, Con: %f, Con(Gr): %f, \n', ...
            iter-1,mse_trainsq(iter-1),mse_testsq(iter-1), consense_err_Sq(iter-1),consense_err_grdSq(iter-1));
        fprintf('          Objective value: %f, Time: %f \n', obj_deSq(iter-1), time_deSq(iter-1));
        fprintf('Iter: %i, Sq (Cen) loss: MSE Train: %f, MSE Test: %f, Obj: %f \n', ...
            iter-1,mse_train_Censq(iter-1),mse_test_Censq(iter-1), obj_ceSq(iter-1));
        fprintf('Iter: %i, Sq (DPG) loss: MSE Train: %f, MSE Test: %f, Con: %f, \n', ...
            iter-1,mse_train_dpg(iter-1),mse_test_dpg(iter-1), consense_err_dpg(iter-1));
        fprintf('          Objective value: %f, Time: %f \n', obj_dpg(iter-1), time_dpg(iter-1));
        
%         fprintf('Duality: %f \n',duality_cen(iter-1));
        toc
    end
    
    %% Distributed Projected Gradient Descent
    gamma_t = 0.1 / ( (iter)^0.5 +1);
    t_s = toc;   
    val_Xbardpg = val_Xdpg*WD; % do the consensus step
    gX = zeros(TRAIN+TEST,N);
    for nn = 1 : N
        % find the individual gradient
        idx_TRAIN_nn = idx_TRAIN( (nn-1)*(TRAIN/N)+1 : nn*TRAIN/N );
        gX(idx_TRAIN_nn,nn) = grad_sq( val_Xbardpg(idx_TRAIN_nn,nn), YTrain_N(idx_TRAIN_nn,nn) );
        [U,D,V] = svd( reshape(val_Xbardpg(:,nn) - N*gamma_t*gX(:,nn), m1, m2 ), 0 );
        dd = diag(D(1:min(m1,m2),1:min(m1,m2)));
        D(1:min(m1,m2),1:min(m1,m2))  = diag( proj_l1(dd,rho) );
        val_Xdpg(:,nn) = vec( U*D*V' );
    end
    t_f = toc;
    
    mse_train_dpg(iter) = max( sum( (repmat(prm_VEC(idx_TRAIN),1,N) - val_Xbardpg(idx_TRAIN,:)).^2 ) ) / TRAIN;
    mse_test_dpg(iter) = max( sum( (repmat(prm_VEC(idx_TEST),1,N) - val_Xbardpg(idx_TEST,:)).^2 ) )  / TEST;
    consense_err_dpg(iter) = max(sum( (val_Xbardpg*ones(N)/N - val_Xbardpg).^2 )) / (TRAIN+TEST);
    time_dpg(iter) = t_f - t_s;
    avg_valXbardpg = val_Xbardpg*ones(N,1)/N;
    obj_dpg(iter) = sum( f_sq( avg_valXbardpg(idx_TRAIN), YTrain(idx_TRAIN) ) );
    
    
    %% Gaussian loss
    gamma_t = 1 / (iter^0.75);
    LT = 1;

    t_s = toc;
    % Decentralized
    if iter == 1
        gX = zeros(TRAIN+TEST,N); % gX is ordered "properly"
        for nn = 1 : N
            idx_TRAIN_nn = idx_TRAIN( (nn-1)*(TRAIN/N)+1 : nn*TRAIN/N );
            gX(idx_TRAIN_nn,nn) = grad_gau( val_Xbar(idx_TRAIN_nn,nn), YTrain_N(idx_TRAIN_nn,nn) );
        end
        gXe = gX;
        gXpGau = gX;
    else
        gX = zeros(TRAIN+TEST,N); % gX is ordered "properly"
        for nn = 1 : N
            idx_TRAIN_nn = idx_TRAIN( (nn-1)*(TRAIN/N)+1 : nn*TRAIN/N );
            gX(idx_TRAIN_nn,nn) = grad_gau( val_Xbar(idx_TRAIN_nn,nn), YTrain_N(idx_TRAIN_nn,nn) );           
        end
        gXe = gXbarGau + gX - gXpGau;
        gXpGau = gX;
    end
    gXbarGau = gXe* (WD^LT);     % aggregate the gradient
    duality_gap = zeros(N,1);
    for nn = 1 : N % the FW step
        [ut,st,vt] = svds(sparse( reshape( gXbarGau(:,nn), m1, m2 ) ),1, 'L', OPT );
        val_add = vec(ut*vt');
        duality_gap(nn) = gXbarGau(idx_TRAIN,nn)'*( val_Xbar(idx_TRAIN,nn) + rho*val_add(idx_TRAIN) );
        val_X(:,nn) = (1-gamma_t)*val_Xbar(:,nn) - gamma_t*rho*val_add;
    end 
    val_Xbar = val_X* (WD^LT); % consensus step
    t_f = toc;
    
    % Eval the MSE (worst case)
    mse_train(iter) = max( sum( (repmat(prm_VEC(idx_TRAIN),1,N) - val_Xbar(idx_TRAIN,:)).^2 ) ) / TRAIN;
    mse_test(iter) = max( sum( (repmat(prm_VEC(idx_TEST),1,N) - val_Xbar(idx_TEST,:)).^2 ) ) / TEST;
    duality(iter) = sum(duality_gap) / N;
    consense_err_Gau(iter) = max(sum( (val_Xbar*ones(N)/N - val_Xbar).^2 )) / (TRAIN+TEST);
    consense_err_grdGau(iter) = max(sum( (gXbarGau*ones(N)/N - gXbarGau).^2 )) / (TRAIN+TEST);
    time_deGau(iter) = t_f - t_s;
    
    avg_valXbar = val_Xbar*ones(N,1)/N;
    % Eval the objective value
    obj_deGau(iter) = sum( f_gau( avg_valXbar(idx_TRAIN), YTrain(idx_TRAIN) ) );
    
    % Centralized
    gX = zeros(TRAIN+TEST,1);
    gX(idx_TRAIN) = grad_gau( val_XCen(idx_TRAIN), YTrain(idx_TRAIN) );
    [ut,st,vt] = svds( sparse( reshape(gX,m1,m2) ), 1 );
    val_add = vec(ut*vt');
    val_XCen = (1-gamma_t)*val_XCen - gamma_t*rho*val_add;
    % Eval the MSE
    mse_train_Cen(iter) =  sum( (prm_VEC(idx_TRAIN) - val_XCen(idx_TRAIN)).^2 )  / TRAIN;
    mse_test_Cen(iter) =  sum( (prm_VEC(idx_TEST) - val_XCen(idx_TEST)).^2 )  / TEST;
    
    obj_ceGau(iter) = sum( f_gau( val_XCen(idx_TRAIN), YTrain(idx_TRAIN) ) );
    
    %% Square Loss
    gamma_t = 2 / (iter+1);
    LT = 1;
    
    t_s = toc;
    % Decentralized
    if iter == 1
        gX = zeros(TRAIN+TEST,N); % gX is ordered "properly"
        for nn = 1 : N
            idx_TRAIN_nn = idx_TRAIN( (nn-1)*(TRAIN/N)+1 : nn*TRAIN/N );
            gX(idx_TRAIN_nn,nn) = grad_sq( val_Xbarsq(idx_TRAIN_nn,nn), YTrain_N(idx_TRAIN_nn,nn) );
        end
        gXe = gX;
        gXpSq = gX;
    else
        gX = zeros(TRAIN+TEST,N); % gX is ordered "properly"
        for nn = 1 : N
            idx_TRAIN_nn = idx_TRAIN( (nn-1)*(TRAIN/N)+1 : nn*TRAIN/N );
            gX(idx_TRAIN_nn,nn) = grad_sq( val_Xbarsq(idx_TRAIN_nn,nn), YTrain_N(idx_TRAIN_nn,nn) );
        end
        gXe = gXbarSq + gX - gXpSq;
        gXpSq = gX;
    end
    gXbarSq = gXe* (WD^LT);     % aggregate the gradient
    
    for nn = 1 : N % the FW step
        [ut,st,vt] = svds(sparse( reshape( gXbarSq(:,nn), m1, m2 ) ),1, 'L', OPT);
        val_add = vec(ut*vt');
        val_Xsq(:,nn) = (1-gamma_t)*val_Xbarsq(:,nn) - gamma_t*rho*val_add;
    end 
    val_Xbarsq = val_Xsq* (WD^LT); % consensus step
    t_f = toc;
    
    % Eval the MSE (worst case)
    mse_trainsq(iter) = max( sum( (repmat(prm_VEC(idx_TRAIN),1,N) - val_Xbarsq(idx_TRAIN,:)).^2 ) ) / TRAIN;
    mse_testsq(iter) = max( sum( (repmat(prm_VEC(idx_TEST),1,N) - val_Xbarsq(idx_TEST,:)).^2 ) )  / TEST;
    
    consense_err_Sq(iter) = max(sum( (val_Xbarsq*ones(N)/N - val_Xbarsq).^2 )) / (TRAIN+TEST);
    consense_err_grdSq(iter) = max(sum( (gXbarSq*ones(N)/N - gXbarSq).^2 )) / (TRAIN+TEST);
    time_deSq(iter) = t_f - t_s;
    
    avg_valXbarsq = val_Xbarsq*ones(N,1)/N;
    obj_deSq(iter) = sum( f_sq( avg_valXbarsq(idx_TRAIN), YTrain(idx_TRAIN) ) );
    
    % Centralized
    gX = zeros(TRAIN+TEST,1);
    gX(idx_TRAIN) = grad_sq( val_XCensq(idx_TRAIN), YTrain(idx_TRAIN) );
    [ut,st,vt] = svds( sparse( reshape(gX,m1,m2) ), 1 );
    val_add = vec(ut*vt');
    val_XCensq = (1-gamma_t)*val_XCensq - gamma_t*rho*val_add;
    % Eval the MSE
    mse_train_Censq(iter) =  sum( (prm_VEC(idx_TRAIN) - val_XCensq(idx_TRAIN)).^2 )  / TRAIN;
    mse_test_Censq(iter) =  sum( (prm_VEC(idx_TEST) - val_XCensq(idx_TEST)).^2 )  / TEST;
    obj_ceSq(iter) = sum( f_sq( val_XCensq(idx_TRAIN), YTrain(idx_TRAIN) ) );

end

semilogy(1:no_iter, obj_deGau/N, 1:no_iter, obj_ceGau/N, 1:no_iter, obj_deSq/N, 1:no_iter, obj_ceSq/N, 1:no_iter, obj_dpg/N );
xlabel('Iteration number'); ylabel('Objective value'); 
legend('DeFW (Gau.)','Cent FW (Gau)','DeFW (Sq.)','Cent FW (Sq.)','DPG (Sq.)');

figure; 
semilogy(1:no_iter, mse_test, 1:no_iter, mse_test_Cen, 1:no_iter, mse_testsq, 1:no_iter, mse_test_Censq, 1:no_iter, mse_test_dpg );
xlabel('Iteration number'); ylabel('Test MSE'); 
legend('DeFW (Gau.)','Cent FW (Gau)','DeFW (Sq.)','Cent FW (Sq.)','DPG (Sq.)');