%%%% ADMM for deep nerual networks training (for approximation task)
%%%% Edited by Jinshan Zeng (Jiangxi Normal University, Email: jsh.zeng@gmail.com)
%%%% Reference: Jinshan Zeng, Shao-Bo Lin, and Yuan Yao, A convergence
%%%% analysis of nonlinearly constrained ADMM in deep learning, JMLR, 2021
% function [W,b,approx_err] = ADMM_DNN(x_train,y_train,W0,b0,act_type,niter)
function [approx_err,trainerr,loss] = ADMM_DNN(x_train,y_train,W0,b0,act_type,niter)
% Inputs:
% x_train, y_train -- input, output of training samples
% W0, b0 -- initializations of weight and bias
% act_type -- activation, 1: sigmoid
% niter -- number of iterations

% Outputs:
% approx_err -- best approximate error during the training
% trainerr -- approximation error during the training
% loss -- the potential function for detecting the convergence behavior


%% Initialization %%%
%%% Using the forward propagation of NNs as the initialization %%%%
L = size(W0,2)-1; % number of hidden layers (excluding the input and output layer)
[~,N_train] = size(x_train);
V0(1).V = act_fun(W0(1).W*x_train + repmat(b0(1).b,1,N_train),act_type);
U0(1).U = zeros(size(V0(1).V,1),size(V0(1).V,2));
for i=2:L
    V0(i).V = act_fun(W0(i).W*V0(i-1).V + repmat(b0(i).b,1,size(V0(i-1).V,2)),act_type);
    U0(i).U = zeros(size(V0(i).V,1),size(V0(i).V,2));
end
V0(L+1).V = W0(L+1).W*V0(L).V + repmat(b0(L+1).b,1,size(V0(L).V,2));
U0(L+1).U = zeros(size(V0(L+1).V,1),size(V0(L+1).V,2));
%%% End initializing procedure %%%

%% implementation of ADMM
% default algorithmic settings
beta = ones(L+1,1); % augmented Lagrangian parameters
lambda = 1e-6; % small regualrization para (default: L2 regularization)

trainerr = zeros(niter,1); % record the approximation error during the training
loss = zeros(niter,1); % record the loss for dectecting the convergence behavior
ntr = size(y_train,2); % number of training samples

% cpu_time = 0;
for k = 1:niter    
    % training accuracy
    ypred_train = NN_output(x_train,W0,b0,act_type);        
    trainerr(k) = norm(ypred_train-y_train)^2/ntr; % training error for regression task
%     tic
    % output layer
    [W(L+1).W,b(L+1).b] = Wout_admm(V0(L).V,V0(L+1).V,U0(L+1).U,beta(L+1),lambda); % update (W,b) at the last layer
    for i=L:-1:2
        [W(i).W,b(i).b] = Whindden_admm(W0(i).W,b0(i).b,V0(i-1).V,V0(i).V,U0(i).U,beta(i),lambda,act_type);
    end
    [W(1).W,b(1).b] = Whindden_admm(W0(1).W,b0(1).b,x_train,V0(1).V,U0(1).U,beta(1),lambda,act_type);
    
    if L==1
        V(1).V = Vhindden_admm(W(1).W,b(1).b,W(2).W,b(2).b,x_train,V0(1).V,V0(2).V,U0(1).U,U0(2).U,beta(1),beta(2),act_type);
        V(L+1).V = Vout_admm(y_train,W(L+1).W,b(L+1).b,V(L).V,U0(L+1).U,beta(L+1));
    elseif L==2
        V(1).V = Vhindden_admm(W(1).W,b(1).b,W(2).W,b(2).b,x_train,V0(1).V,V0(2).V,U0(1).U,U0(2).U,beta(1),beta(2),act_type);
        V(L).V = V2ndend_admm(W(L).W,b(L).b,W(L+1).W,b(L+1).b,V(L-1).V,V0(L+1).V,U0(L).U,U0(L+1).U,beta(L),beta(L+1),act_type); % the 2nd layer from the end
        V(L+1).V = Vout_admm(y_train,W(L+1).W,b(L+1).b,V(L).V,U0(L+1).U,beta(L+1));
    else
        V(1).V = Vhindden_admm(W(1).W,b(1).b,W(2).W,b(2).b,x_train,V0(1).V,V0(2).V,U0(1).U,U0(2).U,beta(1),beta(2),act_type);
        for j=2:L-1
            V(j).V = Vhindden_admm(W(j).W,b(j).b,W(j+1).W,b(j+1).b,V(j-1).V,V0(j).V,V0(j+1).V,U0(j).U,U0(j+1).U,beta(j),beta(j+1),act_type);
        end
        V(L).V = V2ndend_admm(W(L).W,b(L).b,W(L+1).W,b(L+1).b,V(L-1).V,V0(L+1).V,U0(L).U,U0(L+1).U,beta(L),beta(L+1),act_type); % the 2nd layer from the end
        V(L+1).V = Vout_admm(y_train,W(L+1).W,b(L+1).b,V(L).V,U0(L+1).U,beta(L+1));
    end
    
    U(1).U = U0(1).U + beta(1)*(act_fun(W(1).W*x_train+repmat(b(1).b,1,size(x_train,2)),act_type)-V(1).V);
    for i=2:L
        U(i).U = U0(i).U + beta(i)*(act_fun(W(i).W*V(i-1).V+repmat(b(i).b,1,size(V(i-1).V,2)),act_type)-V(i).V);
    end
    U(L+1).U = U0(L+1).U + beta(L+1)*(W(L+1).W*V(L).V+repmat(b(L+1).b,1,size(V(L).V,2))-V(L+1).V);
  
    %% calculating loss for detecting the convergence behavior
    loss(k) = norm(V(L+1).V-y_train,'fro')^2/2+lambda/2*(norm(W(L+1).W,'fro')^2 + norm(W(1).W,'fro')^2 )...
        +beta(L+1)/2*norm(W(L+1).W*V(L).V+repmat(b(L+1).b,1,size(V(L).V,2)) - V(L+1).V,'fro')^2 ...
        +trace((U(L+1).U)'*(W(L+1).W*V(L).V+repmat(b(L+1).b,1,size(V(L).V,2))- V(L+1).V))...
        +beta(1)/2*norm(act_fun(W(1).W*x_train+repmat(b(1).b,1,size(x_train,2)),act_type) - V(1).V,'fro')^2 ...
        +trace((U(1).U)'*(act_fun(W(1).W*x_train+repmat(b(1).b,1,size(x_train,2)),act_type) - V(1).V));
    
    for i=2:L
        temp_loss = lambda/2*norm(W(i).W,'fro')^2 ...
            + beta(i)/2*norm(act_fun(W(i).W*V(i-1).V+repmat(b(i).b,1,size(V(i-1).V,2)),act_type)-V(i).V,'fro')^2 ...
            + trace((U(i).U)'*(act_fun(W(i).W*V(i-1).V+repmat(b(i).b,1,size(V(i-1).V,2)),act_type)-V(i).V));
        loss(k) = loss(k) + temp_loss;
    end
    
    fprintf('epoch: %d, approximation error: %g, augmentedLag: %g \n',k,trainerr(k),loss(k));
        
    %%repeat
    for i=1:L+1
        W0(i).W = W(i).W;
        b0(i).b = b(i).b;
        V0(i).V = V(i).V;
        U0(i).U = U(i).U;
    end
end
approx_err = min(trainerr);
end