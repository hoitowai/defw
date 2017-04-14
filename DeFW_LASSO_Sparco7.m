clear all; close all; clc;

% Simulation code for Fig. 7

no_iter = 5e2;

addpath('./Tools_Data');
% Load the data
load processed_sparco7

rho = 1.1*norm(theta,1); % the constraint param

% construct the decen. mixing
load graph_N50 % to avoid randomness
deg_vec = W*ones(K,1); T1 = repmat(1./deg_vec,1,K); T2 = min(T1,T1').*W;
W_d = diag( ones(K,1) - T2*ones(K,1) ) + T2;
W_extra = (W_d+eye(K))/2;

mean_extreme_obj = zeros(no_iter,1);
mean_extra_obj = zeros(no_iter,1);
mean_rand_obj = zeros(no_iter,1);
mean_dpg_obj = zeros(no_iter,1);
mean_exact_obj = zeros(no_iter,1);

no_comm_total_extreme = mean_extra_obj;
no_comm_total_extra = mean_extra_obj;
no_comm_total_rand = mean_extra_obj;
no_comm_total_dpg = mean_extra_obj;

diff_extreme_theta = mean_extra_obj;
diff_extra_theta = mean_extra_obj;
diff_exact_theta = mean_extra_obj;
diff_dpg_theta = mean_extra_obj;
diff_rand_theta = mean_extra_obj;

% temp variables
no_comm_grad = 0; no_comm_the = 0; no_r_comm_grad = 0;
no_r_comm_the = 0; no_s_comm_the = 0; no_p_comm_the = 0; no_e_comm_the = 0;

% run the DeFW algorithm
theta_fw = zeros(K,n); theta_efw = zeros(K,n); theta_rfw = zeros(K,n);
theta_pg = zeros(K,n);
% we need more mem space for EXTRA
theta_extra = zeros(K,n); old_theta_half = zeros(K,n);


tic;
for fw_iter = 1 : no_iter
    % calculate the gradient 
    dt = min(n,1*ceil(2 + 0.025*(fw_iter)));
    grad_F = zeros(K,n); grad_EF = zeros(K,n);
    grad_RF = zeros(K,n);
    i_union = zeros(n,1); i_r_union = zeros(n,1);
    for k = 1 : K
        grad_F(k,:) = A(:,:,k)'* ( A(:,:,k)*(theta_fw(k,:)') - y(:,k) );
        grad_EF(k,:) = A(:,:,k)'* ( A(:,:,k)*(theta_efw(k,:)') - y(:,k) );
        grad_RF(k,:) = A(:,:,k)'* ( A(:,:,k)*(theta_rfw(k,:)') - y(:,k) );
        % Extreme Co-ord
        [~,sidx] = sort(abs(grad_F(k,:)),'descend');
        i_union(sidx(1:dt)) = 1;
        % Random Co-ord
        sridx = randperm(n,dt);
        i_r_union(sridx) = 1;
    end
    
    Lt = ceil(log(fw_iter) + 1); % the no of GAC used
    
    Agrad_FF = (W_d^(Lt))*grad_F(:,i_union>0); % the truncated GAC (det)!
    Agrad_F = zeros(K,n);
    Agrad_F(:,i_union>0) = Agrad_FF;
    no_comm_grad = no_comm_grad + sum(i_union>0)*Lt;
    
    Agrad_RFF = (W_d^(Lt))*grad_RF(:,i_r_union>0); % the truncated GAC (random)!
    Agrad_RF = zeros(K,n);
    Agrad_RF(:,i_r_union>0) = Agrad_RFF;
    no_r_comm_grad = no_r_comm_grad + sum(i_r_union>0)*Lt;

    Agrad_EF = (W_d^(Lt))*grad_EF; % the exact GAC!
    
    % calculate the FW steps independently
    gamma_t = 1 / (fw_iter+2); tmp_theta_fw = zeros(K,n); 
    tmp_theta_efw = zeros(K,n); tmp_theta_rfw = zeros(K,n);
    for k = 1 : K
        [~,idxk] = max( abs(Agrad_F(k,:) ));
        st = zeros(1,n); st(idxk) = -rho*sign(Agrad_F(k,idxk));
        tmp_theta_fw(k,:) = (1-gamma_t)*theta_fw(k,:)+gamma_t*st;
        
        [~,idxkr] = max( abs(Agrad_RF(k,:) ));
        st = zeros(1,n); st(idxkr) = -rho*sign(Agrad_RF(k,idxkr));
        tmp_theta_rfw(k,:) = (1-gamma_t)*theta_rfw(k,:)+gamma_t*st;
        
        [~,idxke] = max( abs(Agrad_EF(k,:) ));
        st = zeros(1,n); st(idxke) = -rho*sign(Agrad_EF(k,idxke));
        tmp_theta_efw(k,:) = (1-gamma_t)*theta_efw(k,:)+gamma_t*st;
    end
    
    % update theta_fw
    theta_fw = (W_d^(Lt))*tmp_theta_fw;
    theta_efw = (W_d^Lt)*tmp_theta_efw;
    theta_rfw = (W_d^Lt)*tmp_theta_rfw;
    
    no_comm_the = no_comm_the + Lt*sum( ones(1,K)*abs(theta_fw) > 0); % extreme coord case
    no_r_comm_the = no_r_comm_the + Lt*sum( ones(1,K)*abs(theta_rfw) > 0); % random case
    
    % Distributed PG
    alpha_t = 1 / fw_iter;
    bar_theta_pg = W_d*theta_pg;
    for k = 1 : K
        theta_pg(k,:) = proj_l1(bar_theta_pg(k,:) - (alpha_t*A(:,:,k)'* ( A(:,:,k)*(bar_theta_pg(k,:)') - y(:,k)) )' , rho);
    end
    no_p_comm_the = no_p_comm_the + sum( ones(1,K)*abs(theta_pg) > 0 ); % average sparsity

    % PG-EXTRA's distributed grad descent
    alpha = 2000/(n);
    if fw_iter == 1
       old_theta_extra = theta_extra; % keep the old theta..
       cur_bar_theta_extra = W_d*theta_extra; % get the average
       for k = 1 : K
           old_theta_half(k,:) = cur_bar_theta_extra(k,:) - alpha*( A(:,:,k)'* ( A(:,:,k)*(theta_extra(k,:)') - y(:,k)) )';
           theta_extra(k,:) = proj_l1(old_theta_half(k,:),rho);
       end
    else
       tmp_theta_extra = theta_extra; % gonna be the next old_theta...
       cur_bar_theta_extra = W_d*theta_extra; old_bar_theta_extra = W_extra*old_theta_extra;
       for k = 1 : K
           upd1 = cur_bar_theta_extra(k,:) - alpha*( A(:,:,k)'* ( A(:,:,k)*(theta_extra(k,:)') - y(:,k)) )';
           upd2 = old_bar_theta_extra(k,:) - alpha*( A(:,:,k)'* ( A(:,:,k)*(old_theta_extra(k,:)') - y(:,k)) )';
           old_theta_half(k,:) = old_theta_half(k,:) + upd1 - upd2;
           theta_extra(k,:) = proj_l1(old_theta_half(k,:),rho);
       end
       old_theta_extra = tmp_theta_extra;
    end
    no_e_comm_the = no_e_comm_the + 2*sum( ones(1,K)*abs(theta_extra) > 0 ); % average sparsity
    

    % chk the objective
    obj = zeros(K,1); obje = obj; objr = obj; objp = obj; objex = obj; 
    for k = 1 : K
        obj(k) = norm(A(:,:,k)*(theta_fw(k,:)') - y(:,k))^2;
        obje(k) = norm(A(:,:,k)*(theta_efw(k,:)') - y(:,k))^2;
        objr(k) = norm(A(:,:,k)*(theta_rfw(k,:)') - y(:,k))^2;
        objp(k) = norm(A(:,:,k)*(theta_pg(k,:)') - y(:,k))^2;
        objex(k) = norm(A(:,:,k)*(theta_extra(k,:)') - y(:,k))^2;
    end
    mean_extreme_obj(fw_iter) = mean(obj);
    mean_exact_obj(fw_iter) = mean(obje);
    mean_rand_obj(fw_iter) = mean(objr);
    mean_dpg_obj(fw_iter) = mean(objp);
    mean_extra_obj(fw_iter) = mean(objex);
    
    no_comm_total_extreme(fw_iter) = no_comm_the + no_comm_grad;
    no_comm_total_rand(fw_iter) = no_r_comm_the + no_r_comm_grad;
    no_comm_total_dpg(fw_iter) = no_p_comm_the;
    no_comm_total_extra(fw_iter) = no_e_comm_the;
    
    diff_extreme_theta(fw_iter) = norm(theta - theta_fw(1,:)')^2 / norm(theta)^2;
    diff_exact_theta(fw_iter) = norm(theta - theta_efw(1,:)')^2 / norm(theta)^2;
    diff_rand_theta(fw_iter) = norm(theta - theta_rfw(1,:)')^2 / norm(theta)^2;
    diff_dpg_theta(fw_iter) = norm(theta - theta_pg(1,:)')^2 / norm(theta)^2;
    diff_extra_theta(fw_iter) = norm(theta - theta_extra(1,:)')^2 / norm(theta)^2;
    
    if (fw_iter - (20*fix(fw_iter/20)) == 0)
        fprintf('=============== Iteration number: %i ================== \n', fw_iter);
        fprintf('FW (Ext) error: %f, FW (Rand) error: %f, DeFW (exact) error: %f \n',diff_extreme_theta(fw_iter),...
            diff_rand_theta(fw_iter), diff_exact_theta(fw_iter));
        fprintf('FW (Ext) obj: %f, FW (Rand) obj: %f, DeFW (exact) obj: %f \n',mean_extreme_obj(fw_iter),...
            mean_rand_obj(fw_iter), mean_exact_obj(fw_iter));
        fprintf('D-PG error: %f, EXTRA error: %f \n',diff_dpg_theta(fw_iter),diff_extra_theta(fw_iter));
        fprintf('FW (Ext) Comm: %i, FW (Rand) Comm: %i, EXTRA comm %i  , D-PG comm: %i \n',no_comm_total_extreme(fw_iter),...
            no_comm_total_rand(fw_iter),no_comm_total_extra(fw_iter), no_comm_total_dpg(fw_iter));
        toc;
    end
end

loglog( no_comm_total_extreme, mean_extreme_obj, non_comm_total_rand, mean_rand_obj, ...
    no_comm_total_dpg, mean_dpg_obj, no_comm_total_extra, mean_extra_obj );
xlim([1e4,1e7]);
legend('DeFW (extreme)','DeFW (rand)','DPG','PG-EXTRA');
xlabel('Communication Cost'); ylabel('Objective value');