function Vstar = updateV_inner_bcd(U1,U2,W,b,Vprev,gamma,act_type,opt_alg)
% update V for the inner layer
% minimization problem
% min_V gamma/2*||V-sigma(U1)||_F^2 + rho/2*|U2-W*V-b|||_F^2
% Vprev: previous V
% alpha: the inverse of step size
% act_type: activation type
[~,d] = size(W);
I = sparse(eye(d));
U1 = act_fun(U1,act_type);
switch opt_alg
    case 1 % exact solution
        Vstar = (gamma*(W'*W)+gamma*I)\(gamma*W'*(U2-repmat(b,1,size(U2,2)))+gamma*U1);
    case 2 % gradient descent
        maxiter = 20;
        alpha = gamma + gamma*norm(W,'fro')^2;
        for i=1:maxiter
            Vstar = Vprev - (gamma*(Vprev-U1)+gamma*W'*(W*Vprev + repmat(b,1,size(U2,2))-U2))/alpha;
            Vprev = Vstar;
        end
end
end