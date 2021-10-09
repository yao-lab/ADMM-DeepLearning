%% update weight and bias of the hidden layers except the last layer
function [W,b] = Whindden_admm(W1prev,b1prev,V0prev,V1prev,U1prev,beta,lambda,act_type)
% input:
% W1prev -- the previous update of W
% b1prev -- the previous update of b
% V0prev -- the previous update of V at the previous layer
% V1prev -- the previous update of V at the current layer
% U1prev -- the previous update of multiplier at the current layer
% beta -- augmented Lagrangian parameter at the current layer
% lambda -- regularization parameter
% act_type -- activation type, 1: sigmoid; 2: ReLU

W1prev = [W1prev b1prev];
V0prev = [V0prev; ones(1,size(V0prev,2))];
W_total = aux_Wb(W1prev,V0prev,V1prev-U1prev/beta,beta,lambda,act_type);
W = W_total(:,1:size(W_total,2)-1);
b = W_total(:,end);
clear W_total Wprev V0prev;
end


%% auxilary subproblem for updating W
%W = arg min{lambda/2*||W||_F^2 + beta*H_{sigma}^k(W:A,B)}
function W = aux_Wb(Wprev,A,B,beta,lambda,act_type)
% input:
% Wprev -- the previous update of W
% A,B -- auxilarity matrices
I = sparse(eye(size(A,1))); % indentity matrix
h = max(max(abs(B))); % the inverse of step size
temp = Wprev*A;
G = (act_fun(temp,act_type)-B).*act_fun_Grad(temp,act_type);
W = (beta*(h*temp/2 - G)*(A'))/(lambda*I+beta*h*A*(A')/2);
clear temp G I h;
end