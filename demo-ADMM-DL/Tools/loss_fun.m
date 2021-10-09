function f = loss_fun(x,y,loss_type)
% input:
% x -- predict of data
% y -- output of data
% loss_type -- 1: least square; 2: hinge
switch loss_type
    case 1 % least square
        f = sum(sum((x-y).^2,1))/2;
    case 2 % hinge loss
        f = sum(max(0,1-sum(x.*y,2)))/2;
end