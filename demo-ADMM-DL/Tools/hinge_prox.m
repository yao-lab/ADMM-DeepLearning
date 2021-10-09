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