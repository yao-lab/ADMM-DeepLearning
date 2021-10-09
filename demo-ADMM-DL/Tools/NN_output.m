function out = NN_output(X,W,b,act_type)
% output for neural networks with L-layers including L-2 hidden layers
% X -- input of training data, d_in*N
% W -- weight matrices W = [W1, W2, ..., W(L-1)]
% b -- bias vectors b = [b1, b2, ..., b(L-1)]
% act_type -- activation function, 1: sigmoid, 2: ReLU
L = size(W,2)+1;
a(1).a = X;
n = size(X,2); % sample size
for j=1:L-2
   z(j).z = W(j).W*a(j).a + repmat(b(j).b,1,n);
   a(j+1).a = act_fun(z(j).z, act_type);
end
out = W(L-1).W*a(L-1).a + repmat(b(L-1).b,1,n); % using linear activation at the last layer
end