%% update V of the hidden layers except the last layer and the 2nd layer from the end
function V = Vhindden_admm(W1,b1,W2,b2,V0,V1prev,V2prev,U1prev,U2prev,beta1,beta2,act_type)
% input:
% W1 b1 -- the current update of (W,b) at the cuurent layer
% W2 b2 -- the current update of (W,b) at the latter layer
% V0 -- the current update of V at the previous layer
% V1prev -- the previous update of V at the current layer
% V2prev -- the previous update of V at the latter layer
% U1prev -- the previous update of multiplier at the current layer
% U2prev -- the previous update of multiplier at the latter layer
% beta1 -- augmented Lagrangian parameter at the current layer
% beta2 -- augmented Lagrangian parameter at the latter layer
% act_type -- activation type, 1: sigmoid; 2: ReLU

B = V2prev-U2prev/beta2;
C = act_fun(W1*V0+repmat(b1,1,size(V0,2)),act_type) + U1prev/beta1;
V = aux_V(V1prev,W2,b2,B,C,beta1,beta2,act_type);
clear B C;
end


%% auxilary subproblem for updating V
%V = arg min{beta1/2*||V-C||_F^2 + beta2*M_{sigma}^k(V:W2,B)}
function V = aux_V(Vprev,W2,b2,B,C,beta1,beta2,act_type)
% input:
% W2, b2 -- the current update of (W,b) at the latter layer
% Vprev -- the previous update of V
% B,C -- auxilarity matrices
temp = W2*Vprev;
I = sparse(eye(size(W2,2)));
mu = max(max(abs(B)));
hidden = temp+repmat(b2,1,size(Vprev,2));
G = (act_fun(hidden,act_type)-B).*act_fun_Grad(hidden,act_type); % gradient
V = (beta1*I+beta2*mu*(W2')*W2/2)\(beta1*C+beta2*(W2')*(mu*temp/2-G));
clear temp I mu hidden G;
end