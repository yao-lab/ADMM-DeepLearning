function [W0,b0] = init_scheme(dim,init_type,scal_factor)
% initialization schemes
% dim: neural network structures
% init_type: 0: zero-init, 1: Orth-Unif, 2: Orth-Gauss, 3: LeCun-Unif,
% 4: LeCun-Gauss, 5: Xavier, 6: MSRA
% scaling factor: usually for MSRA, scal_factor = range.^(-1/number of layers), where range is the range of the input, i.e., (-range, range)
if size(dim,1)<2
    disp('please define a suitable deep neural network with at least one hidden layer');
    return;
else
    L = size(dim,1)-2; % number of hidden layers
    switch init_type
        case 0 % all-zero initialization
            for layer=1:L+1
                W0(layer).W = zeros(dim(layer+1),dim(layer));
                b0(layer).b = zeros(dim(layer+1),1);
            end
        case 1 % random orthogonal initialization with uniforma distribution
            for layer=1:L+1
                W0(layer).W = -1+2*rand(dim(layer+1),dim(layer));
                if dim(layer+1)<dim(layer) % W0*W0'=I
                    tempW = orth((W0(layer).W).');
                    W0(layer).W = scal_factor*tempW.';
                else % W0'*W0=I
                    W0(layer).W =scal_factor*orth(W0(layer).W);
                end
                b0(layer).b = zeros(dim(layer+1),1);
            end
        case 2 % random orthogonal initialization with Gaussian distribution
            for layer=1:L+1
                W0(layer).W = randn(dim(layer+1),dim(layer));
                if dim(layer+1)<dim(layer) % W0*W0'=I
                    tempW = orth((W0(layer).W).');
                    W0(layer).W = scal_factor*tempW.';
                else % W0'*W0=I
                    W0(layer).W =scal_factor*orth(W0(layer).W);
                end
                b0(layer).b = zeros(dim(layer+1),1);
            end
        case 3 % LeCun random oinitialization with uniforma distribution
            for layer=1:L+1
                W0(layer).W = scal_factor*(-1+2*rand(dim(layer+1),dim(layer)))*sqrt(3/dim(layer));
                b0(layer).b = zeros(dim(layer+1),1);
            end
        case 4 % LeCun random oinitialization with Gaussian distribution
            for layer=1:L+1
                W0(layer).W = scal_factor*randn(dim(layer+1),dim(layer))*sqrt(1/dim(layer));
                b0(layer).b = zeros(dim(layer+1),1);
            end
        case 5 % Xavier initialization
            for layer=1:L+1
                W0(layer).W = scal_factor*(-1+2*rand(dim(layer+1),dim(layer)))*sqrt(6/(dim(layer+1)+dim(layer)));
                b0(layer).b = zeros(dim(layer+1),1);
            end
        case 6 % MSRA initialization
            % scaling factor for data with z-scoring normalization
%             scal_factor = dim(1).^(-1/(L+1)); % to make the products of all layers approach to 1
            for layer=1:L
                W0(layer).W = scal_factor*randn(dim(layer+1),dim(layer))*sqrt(2/dim(layer+1));
                b0(layer).b = zeros(dim(layer+1),1);                
            end
            W0(L+1).W = scal_factor*randn(dim(L+2),dim(L+1))*sqrt(1/dim(L+2));
            b0(L+1).b = zeros(dim(L+2),1);
    end
    clear layer;
end
end