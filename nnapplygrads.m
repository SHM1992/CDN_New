function net = nnapplygrads(net)
    %  TODO add momentum
%    for i = 1 : (numel(net.size) - 1)
    for i = 1 : (net.n - 1)
                net.W{1} = net.W{1} - net.alpha * (net.dW{1} + net.lambda * net.W{1});
                net.b{1} = net.b{1} - net.alpha * net.db{1};
    end
end
