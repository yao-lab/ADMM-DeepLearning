% update V at the second layer from the end 
function V = V2ndend_admm(W1,b1,W2,b2,V0,V2prev,U1prev,U2prev,beta1,beta2,act_type)
% input:
% W1, b1 -- the current update of weight and bias at the current layer
% W2, b2 -- the current update of weight and bias at the latter layer
% V0 -- the current update V at the previous one layer
% V1prev -- the previous update of V at the current layer
% V2prev -- the previous update of V at the latter layer
% U1prev -- the previous update of multiplier at the current layer
% U2prev -- the previous update of multiplier at the latter layer
% beta1 -- augmented Lagrangian parameter at the current layer
% beta2 -- augmented Lagrangian parameter at the latter layer
temp = U1prev + beta1*act_fun(W1*V0+repmat(b1,1,size(V0,2)),act_type) - (W2')*(U2prev-beta2*(V2prev-repmat(b2,1,size(V2prev,2))));
I = sparse(eye(size(W2,2)));
V = (beta1*I+beta2*(W2')*W2)\temp;
clear temp I;
end