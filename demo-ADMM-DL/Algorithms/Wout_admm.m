%% update the weight and bias of the last layer
% WL = [W b]
% V_{L-1} = [V_{L-1}; ones(1,size(V_{L-1},2))]
% WL = (beta*VLprev - ULprev)*V_{L-1}prev^T*inv(lambda*I+beta*V_{L-1}prev*V_{L-1}prev^TV)
function [W,b] = Wout_admm(V0prev,V1prev,U1prev,beta,lambda)
% input:
% V0prev -- V_{L-1}^{k-1}, the previous update of hidden state V at the previous layer
% V1prev -- VL^{k-1}, the previous update of hidden state V at the current layer
% U1prev -- UL^{k-1}, the previous update of multiplier U at the current layer
% beta -- augmented Lagrangian at the current layer
% lambda -- regularization parameter

[m,n] = size(V0prev);
V0_total = [V0prev; ones(1,n)];
I = sparse(eye(m+1));
A = (beta*V1prev-U1prev)*(V0_total');
B = lambda*I+beta*V0_total*(V0_total');
W_total = A/B;
W = W_total(:,1:size(W_total,2)-1);
b = W_total(:,end);
clear A B I V0_total W_total;
end