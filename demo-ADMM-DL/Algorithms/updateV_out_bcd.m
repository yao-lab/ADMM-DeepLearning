% V-subproblem: min_V Risk(V;Y)+gamma/2 ||V-U||_F^2 + alpha/2 ||V-Vprev||_F^2
% the first term Risk: the empirical loss
% the second term: the penalty term
% the last term: a proximal term 
function Vstar = updateV_out_bcd(Vprev,U,Y,gamma,alpha,loss_type)
% update of the variable in the output layer
% Vprev: the previous V in the output layer
% U: the update of U in the output layer
% Y: the output label
% gamma: the penalty parameter
% alpha: the inverse of step size (or proximal parameter), (alpha =0 if the loss function is l2)
% loss_type-- 1: least square; 2: hinge
%%%% opt_alg: optimization algorithms, 1=exact solution; 2 = gradient type
switch loss_type
    case 1 % least square--exact solve (alpha = 0)
        alpha = 0;
        Vstar = (Y+gamma*U+alpha*Vprev)/(1+gamma+alpha);
    case 2 % hinge loss -- exact solve 
        temp = gamma+alpha;
        Yprev = gamma/temp*U + alpha/temp*Vprev;
        Vstar = hinge_prox(Y,Yprev,temp);
end
end


% proximal operator for the hinge-min
function z = hinge_prox(a,b,gamma)
% hinge_prox(a,b,gamma) = argmin_u max{0,1-a*u} + gamma/2 (u-b)^2
% a: m*1 vector
% b: m*1 vector
% gamma>0: parameter
% z: output of the proximal of hinge
% m = size(a,1);
z = b.*(a==0)...
    + (a~=0).*((b+gamma^(-1)*a).*(a.*b<=1-gamma^(-1)*a.*a)...
    + (a.^(-1)).*(a.*b>1-gamma^(-1)*a.*a&a.*b<1)...
    + b.*(a.*b>=1));
end