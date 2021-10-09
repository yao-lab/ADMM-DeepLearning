function U = updateU_inner_bcd(Uprev,V0,V1,W,b,gamma,alpha,act_type)
% min gamma/2 ||V1-act(U)||_F^2 + gamma/2*||U-W*V0-b||_F^2 + alpha/2*||U-Uprev||_F^2
% act_type -- 1: sigmoid; 2: ReLU
% update for the inner layer of U
[~,N] = size(Uprev);
switch act_type
    case 1 % sigmoid (using gradient descent)
        maxiter = 20;        
        for iter = 1:maxiter
           U = Uprev - alpha*gamma*((act_fun(Uprev,act_type)-V1).*act_fun_Grad(Uprev,act_type)+(Uprev-W*V0-repmat(b,1,N)));
           Uprev = U;
        end
    case 2 % ReLU (using proximal)
        temp = gamma + alpha;
        tempU = gamma/temp*(W*V0+repmat(b,1,N))+alpha/temp*Uprev;
        U = relu_prox(V1,tempU,temp/gamma);
end
end

function val = relu_prox(a,b,gamma)
% relu-prox
% u*=arg min (ReLU(u)-a)^2/2 + gamma*(u-b)^2/2
    [d,N] = size(a);
    val = zeros(d,N);
    x = (a+gamma*b)/(1+gamma);
    y = min(b,0);
	val((a+gamma*b >= 0 & b >=0) | (a*(gamma-sqrt(gamma*(gamma+1))) <= gamma*b & b < 0)) ...
        = x((a+gamma*b >= 0 & b >=0) | (a*(gamma-sqrt(gamma*(gamma+1))) <= gamma*b & b < 0));
    
    val(-a <= gamma*b & gamma*b <= a*(gamma-sqrt(gamma*(gamma+1)))) ...
        = b(-a <= gamma*b & gamma*b <= a*(gamma-sqrt(gamma*(gamma+1))));
    
	val(a+gamma*b < 0) = y(a+gamma*b < 0);
end