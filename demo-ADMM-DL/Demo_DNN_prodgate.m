%%%% On realizing product-gate function via ADMM, please cite the following paper:
% [1] Jinshan Zeng, Shao-Bo Lin, Yuan Yao, Ding-Xuan Zhou, On ADMM in deep learning: Convergence and Saturation-avoidance, Journal of Machine Learning Research, 2021
% edited by Jinshan Zeng (Jiangxi Normal University, jsh.zeng@gmail.com)
clear all;
close all;
clc;

addpath Algorithms Tools;

rng('default');
seed = 30;
rng(seed);
fprintf('Seed = %d \n', seed)


%% generate training data
ntr = 1000; % number of training samples
d = 2; % input dimension
xtr = -1+2*rand(ntr,2).'; % input lying in [-1,1]
ytr = xtr(1,:).*xtr(2,:); % output y=x1*x2


% neural network structure
% deep fully connected nets are considered for simplicity
L = 2; % number of hidden layers (excluding the input and output layer)
dim = zeros(L+2,1); % dimensions of each layer
dim(1) = size(xtr,1); % input dimension
dim(L+2) = size(ytr,1); % output dimension
for i=2:L+1
    dim(i) =300; % the number of hidden neurons 
end
clear i;

%% initialization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[W0,b0] = init_scheme(dim,6,8); % He kaiming initialization
loss_type = 1; % least square
acttype = 1; % 1: sigmoid; 2: ReLU; 3: leaky ReLU; 4: linear; 5: sign

%% Implement different algorithms %%%%%%%%%%%%%%%%%%%%
NumEpoch = 30; % number of epochs
disp('starting ADMM for sigmoid type DNNs');

tic;
[ApproxErr,trainerr,loss] = ADMM_DNN(xtr,ytr,W0,b0,acttype,NumEpoch);
cputime = toc;

fprintf('approximation error: %g, running time: %g \n',ApproxErr,cputime);
figure,
Epoch = (1:NumEpoch)';
semilogy(Epoch,trainerr,'r-','LineWidth',2);
xlabel('Epoch number','FontSize',18);
ylabel('Approximation error','FontSize',18);

figure,
semilogy(Epoch,loss,'k-','LineWidth',2);
xlabel('Epoch number','FontSize',18);
ylabel('Potential function','FontSize',18);