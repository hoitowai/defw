m1 = 200; m2 = 1000; rankr = 5;
prm_bar = zeros(m1,m2);
for rr = 1 : rankr
    prm_bar = prm_bar + randn(m1,1)*randn(1,m2) / rankr;
end

TRAIN = 4e4; TEST = 16e4;
YTrain = zeros(m1*m2,1);
idx_TRAIN = randperm(m1*m2,TRAIN);
YTrain( idx_TRAIN ) = prm_bar( idx_TRAIN );
% these are the global ones

N = 50; 
YTrain_N = zeros(m1*m2,N); YTrain_N_noiseless = YTrain_N;
for nn = 1 : N
    YTrain_N(idx_TRAIN((nn-1)*(TRAIN/N)+1 : nn*TRAIN/N),nn) = ...
        prm_bar( idx_TRAIN((nn-1)*(TRAIN/N)+1 : nn*TRAIN/N) )' + ...
        5* (rand(TRAIN/N,1)<0.2).*randn(TRAIN/N,1);
    YTrain_N_noiseless(idx_TRAIN((nn-1)*(TRAIN/N)+1 : nn*TRAIN/N),nn) = ...
        prm_bar( idx_TRAIN((nn-1)*(TRAIN/N)+1 : nn*TRAIN/N) );
end
% these are the locals

prm_VEC = vec(prm_bar);
idx_TEST = setdiff( 1:m1*m2, idx_TRAIN);
YTrain_noiseless = YTrain;
YTrain = YTrain_N*ones(N,1);