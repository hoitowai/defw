clear all; close all; clc;
% MovieLens Data
load ML100k

m1 = 943; m2 = 1682;
sigma = 5;
rho = 10000;

% define the gradient
grad_sq = @(x,y) (x-y);
f_sq = @(x,y) (x-y).^2;
grad_gau = @(x,y) (x-y).*exp(-(x-y).^2 / (2*sigma)) / sigma;
f_gau = @(x,y) 1 - exp( -(x-y).^2 / (2*sigma));

TRAIN = 80e3; TEST = 20e3;

% convert the data into the "list" format
[rows,cols,val] = find(YTrain);
[rows_t,cols_t,val_t] = find(YTest); 
rows_FULL = [rows;rows_t]; cols_FULL = [cols; cols_t]; val_FULL = [val; val_t];
YFull = spconvert([rows_FULL,cols_FULL,val_FULL]);

% construct the masks for easy calculation
MASKTrain = spconvert([rows,cols,ones(80e3,1)]);
MASKTest = spconvert([rows_t,cols_t,ones(20e3,1); rows, cols, zeros(80e3,1)]);

% construct the permutation matrix for nonzeros function
% order = nonzeros( (MASKTest+MASKTrain).* reshape( 1:m1*m2, m1, m2 ) );
% idx_perm = zeros(TRAIN+TEST,1);
% for ii = 1 : (TRAIN+TEST)
%     idx_perm(ii) = find( order == (cols_FULL(ii)-1)*m1 + rows_FULL(ii) );
% end
load nonzeros_perm

val_corrupted = val + 5*(rand(TRAIN,1)<0.2).*randn(TRAIN,1); % the corrupted training data

YTrain_Bad = spconvert([rows,cols,val_corrupted]);


N = 50; 
% Create the observations for each agent
rows_N = zeros(TRAIN,N); cols_N = zeros(TRAIN,N); val_corrupted_N = zeros(TRAIN,N);
MASK_N = zeros(TRAIN,N);
for nn = 1 : N
    MASK_N( (nn-1)*(TRAIN/N)+1 : nn*(TRAIN/N) , nn) = 1;
    rows_N( (nn-1)*(TRAIN/N)+1 : nn*(TRAIN/N) , nn) = rows( (nn-1)*(TRAIN/N)+1 : nn*(TRAIN/N) );
    cols_N( (nn-1)*(TRAIN/N)+1 : nn*(TRAIN/N) , nn) = cols( (nn-1)*(TRAIN/N)+1 : nn*(TRAIN/N) );
    val_corrupted_N( (nn-1)*(TRAIN/N)+1 : nn*(TRAIN/N) , nn) = ...
        val_corrupted( (nn-1)*(TRAIN/N)+1 : nn*(TRAIN/N) );
end

% Create the initial matrices for each agent [need only the stuffs from FULL]
val_X = zeros(TRAIN+TEST,N); val_Xbar = val_X; % for Decen+Gau
val_XCen = zeros(TRAIN+TEST,1); % for Cen+Gau
val_Xsq = zeros(TRAIN+TEST,N); val_Xbarsq = val_Xsq; % for Decen+Sq
val_XCensq = zeros(TRAIN+TEST,1);

% construct the decen. mixing matrix (make it static first)
% pc = 0.1;
% W = (rand(N)<pc); W = W + eye(N); W = W + W'; W = (W>0);
load graph_N50 % to avoid randomness
deg_vec = W*ones(N,1); T1 = repmat(1./deg_vec,1,N); T2 = min(T1,T1').*W;
WD = diag( ones(N,1) - T2*ones(N,1) ) + T2;

no_iter = 10e3;
mse_train = zeros(no_iter,1); mse_test = zeros(no_iter,1); duality = zeros(no_iter,1);
mse_train_Cen = zeros(no_iter,1); mse_test_Cen = zeros(no_iter,1);
mse_trainsq = zeros(no_iter,1); mse_testsq = zeros(no_iter,1);
mse_train_Censq = zeros(no_iter,1); mse_test_Censq = zeros(no_iter,1);

consense_err_Gau = zeros(no_iter,1); consense_err_Sq = zeros(no_iter,1);
consense_err_grdGau = zeros(no_iter,1); consense_err_grdSq = zeros(no_iter,1);

tic;
for iter = 1 : no_iter
    
    if mod(iter,10) == 2
        fprintf('Iter: %i, Gau (Decen) loss: MSE Train: %f, MSE Test: %f, Duality: %f, Cons: %f, Cons(Grd): %f \n', ...
            iter-1,mse_train(iter-1),mse_test(iter-1),duality(iter-1),consense_err_Gau(iter-1),consense_err_grdGau(iter-1));
        fprintf('Iter: %i, Gau (Cen) loss: MSE Train: %f, MSE Test: %f \n', ...
            iter-1,mse_train_Cen(iter-1),mse_test_Cen(iter-1));
        fprintf('Iter: %i, Sq (Decen) loss: MSE Train: %f, MSE Test: %f, Cons: %f, Cons(Grd): %f \n', ...
            iter-1,mse_trainsq(iter-1),mse_testsq(iter-1),consense_err_Sq(iter-1),consense_err_grdSq(iter-1));
        fprintf('Iter: %i, Sq (Cen) loss: MSE Train: %f, MSE Test: %f \n', ...
            iter-1,mse_train_Censq(iter-1),mse_test_Censq(iter-1));
        toc
    end
    
    %% Gaussian loss
    gamma_t = 1 / (iter^0.5);
    % Controls the number of GAC updates per iteration
    LT = 1;
    
    % Decentralized
    if iter == 1
        gX = zeros(TRAIN,N);
        for nn = 1 : N
            gX(:,nn) = MASK_N(:,nn).*grad_gau( val_Xbar(1:TRAIN,nn), val_corrupted_N(:,nn) );
        end
        gXe = gX;
        gXpGau = gX;
    else
        gX = zeros(TRAIN,N);
        for nn = 1 : N
            gX(:,nn) = MASK_N(:,nn).*grad_gau( val_Xbar(1:TRAIN,nn), val_corrupted_N(:,nn) );
        end
        gXe = gXbarGau + gX - gXpGau;
        gXpGau = gX;
    end
    gXbarGau = gXe* (WD^LT);     % aggregate the gradient
    duality_gap = zeros(N,1);
    tic;
    for nn = 1 : N % the FW step
        [ut,st,vt] = svds(spconvert([rows,cols,gXbarGau(:,nn)]),1);
        val_add = nonzeros( (MASKTest+MASKTrain).*( ut*vt' ) + (MASKTest+MASKTrain) ) - 1;
        val_add = val_add(idx_perm);
        duality_gap(nn) = gXbarGau(:,nn)'*( val_Xbar(1:TRAIN,nn) + rho*val_add(1:TRAIN) );
        val_X(:,nn) = (1-gamma_t)*val_Xbar(:,nn) - gamma_t*rho*val_add;
    end

    val_Xbar = val_X* (WD^LT); % consensus step
    % Eval the MSE (worst case)
    mse_train(iter) = max( sum( (repmat(val,1,N) - val_Xbar(1:TRAIN,:)).^2 ) ) / TRAIN;
    mse_test(iter) = max( sum( (repmat(val_t,1,N) - val_Xbar(TRAIN+1:end,:)).^2 ) ) / TEST;
    duality(iter) = sum(duality_gap) / N;
    
    consense_err_Gau(iter) = max(sum( (val_Xbar*ones(N)/N - val_Xbar).^2 )) / (TRAIN+TEST);
    consense_err_grdGau(iter) = max(sum( (gXbarGau*ones(N)/N - gXbarGau).^2 )) / (TRAIN);
    
    % Centralized
    gX = grad_gau( val_XCen(1:TRAIN), val_corrupted );
    [ut,st,vt] = svds( spconvert([rows,cols,gX]), 1 );
    val_add = nonzeros( (MASKTest+MASKTrain).* (ut*vt') + (MASKTest+MASKTrain) ) - 1;
    val_XCen = (1-gamma_t)*val_XCen - gamma_t*rho*val_add(idx_perm);
    % Eval the MSE
    mse_train_Cen(iter) =  sum( (val - val_XCen(1:TRAIN)).^2 )  / TRAIN;
    mse_test_Cen(iter) =  sum( (val_t - val_XCen(TRAIN+1:end)).^2 )  / TEST;
    
    %% Square Loss
    gamma_t = 2 / (iter+1);
    % Controls the no of GAC updates per iteration
    LT = 1;

    % Decentralized
    if iter == 1
        gX = zeros(TRAIN,N);
        for nn = 1 : N
            gX(:,nn) = MASK_N(:,nn).*grad_sq( val_Xbarsq(1:TRAIN,nn), val_corrupted_N(:,nn) );
        end
        gXe = gX;
        gXpSq = gX;
    else
        gX = zeros(TRAIN,N);
        for nn = 1 : N
            gX(:,nn) = MASK_N(:,nn).*grad_sq( val_Xbarsq(1:TRAIN,nn), val_corrupted_N(:,nn) );
        end
        gXe = gXbarSq + gX - gXpSq;
        gXpSq = gX;
    end
    gXbarSq = gXe* (WD^LT);     % aggregate the gradient
    for nn = 1 : N % the FW step
        [ut,st,vt] = svds(spconvert([rows,cols,gXbarSq(:,nn)]),1);
        val_add = nonzeros( (MASKTest+MASKTrain).*( ut*vt' ) + (MASKTest+MASKTrain) ) - 1;
        val_Xsq(:,nn) = (1-gamma_t)*val_Xbarsq(:,nn) - gamma_t*rho*val_add(idx_perm);
    end 
    val_Xbarsq = val_Xsq* (WD^LT); % consensus step
    % Eval the MSE (worst case)
    mse_trainsq(iter) = max( sum( (repmat(val,1,N) - val_Xbarsq(1:TRAIN,:)).^2 ) ) / TRAIN;
    mse_testsq(iter) = max( sum( (repmat(val_t,1,N) - val_Xbarsq(TRAIN+1:end,:)).^2 ) ) / TEST;
    
    consense_err_Sq(iter) = max(sum( (val_Xbarsq*ones(N)/N - val_Xbarsq).^2 )) / (TRAIN+TEST);
    consense_err_grdSq(iter) = max(sum( (gXbarSq*ones(N)/N - gXbarSq).^2 )) / (TRAIN);
    
    % Centralized
    gX = grad_sq( val_XCensq(1:TRAIN), val_corrupted );
    [ut,st,vt] = svds( spconvert([rows,cols,gX]), 1 );
    val_add = nonzeros( (MASKTest+MASKTrain).* (ut*vt') + (MASKTest+MASKTrain) ) - 1;
    val_XCensq = (1-gamma_t)*val_XCensq - gamma_t*rho*val_add(idx_perm);
    % Eval the MSE
    mse_train_Censq(iter) =  sum( (val - val_XCensq(1:TRAIN)).^2 )  / TRAIN;
    mse_test_Censq(iter) =  sum( (val_t - val_XCensq(TRAIN+1:end)).^2 )  / TEST;    
    
end

figure; 
semilogy(1:no_iter, mse_test, 1:no_iter, mse_test_Cen, 1:no_iter, mse_testsq, 1:no_iter, mse_test_Censq );
xlabel('Iteration number'); ylabel('Test MSE'); 
legend('DeFW (Gau.)','Cent FW (Gau)','DeFW (Sq.)','Cent FW (Sq.)');