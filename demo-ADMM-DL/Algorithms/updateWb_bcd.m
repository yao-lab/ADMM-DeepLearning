% update (W,b) 
% minimization problem:
% min_(W,b) ...
% gamma/2*||U-WV-b||_F^2+alpha/2*||W-Wprev||_F^2+alpha/2*||b-bprev||_2^2 + lambda/2*\|W\|_F^2
% gamma: the penalty parameter
% alpha: the inverse of step size (or, the proximal parameter) 
% alpha >= rho, say, alpha = rho*N+1
% lambda: regularization para.
function [Wstar,bstar] = updateWb_bcd(U,V,Wprev,bprev,gamma,alpha,lambda,opt_alg)
% update for the inner layers
% input: U, V
% Wprev: the previous W
% bprev: the previous b
% opt_alg: optimization algorithm (1: exact solution, 2: gradient descent)
[d,N] = size(V);
switch opt_alg
    case 1 % exact solution    
    I = sparse(eye(d)); 
    Wstar = (alpha*Wprev+gamma*(U-repmat(bprev,1,size(U,2)))*V')/((alpha+lambda)*I+gamma*(V*V'));
    bstar = (alpha*bprev+gamma*sum(U-Wprev*V,2))/(gamma*N+alpha);
    case 2 % gradient descent
        maxiter =20;
        for i=1:maxiter
            Wstar = Wprev - (gamma*(Wprev*V+repmat(bprev,1,size(U,2))-U)*V'+lambda*Wprev)/alpha;
            bstar = bprev - gamma*N*(bprev+Wstar*mean(V,2)-mean(U,2))/alpha;
            Wprev = Wstar;
            bprev = bstar;
        end
end
end