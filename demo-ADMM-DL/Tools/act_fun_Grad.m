function f = act_fun_Grad(v,act_type)
% grad of activation function
% act_type, 1: sigmoid; 2: ReLU
switch act_type
    case 1 % sigmoid activation
        z = act_fun(v,act_type);
        f = z.*(1-z); 
    case 2 % ReLU
        f = (v>0);
    case 3 % leaky ReLU
        a=0.5;
        f = (v>0) + a*(1-(v>0));
    case 4 % linear
        f = 1;
end
end