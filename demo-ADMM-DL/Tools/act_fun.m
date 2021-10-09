function f = act_fun(u,act_type)
% activation function
% act_type, 1: sigmoid; 2: ReLU; 3: leaky ReLU; 4: linear; 5: sign
switch act_type
    case 1 % sigmoid activation
        f = 1./(1+exp(-u)); 
    case 2 % ReLU
        f = max(0,u);
    case 3 % leaky ReLU
        a = 0.5;
        f = u.*(u>0)+a*u.*(1-(u>0));
    case 4 % linear
        f = u;
    case 5 % sign (binary)
        f = sign(u);
end
end