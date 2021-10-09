%% update V at the last layer
% VL = [Y+ULprev+betaL(WLV_{L-1}+bL)]/(1+beta)
function V = Vout_admm(Y,W1,b1,V0,U1prev,beta)
% input:
% Y -- label
% W1, b1 -- the current update of weight and bias
% V0 -- the current update of V at the previous one layer
% U1prev -- the previous update of multiplier at the current layer
V = (Y+U1prev + beta*(W1*V0 + repmat(b1,1,size(V0,2))))/(1+beta);
end