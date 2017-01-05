function net = nnff(net, x, y)
%    n = numel(net.size);
    n = net.n;
    m = size(x, 1);

    net.a{1} = x;

    %%  feedforward pass
    net.p{1} = 0.99 * net.p{1} + 0.01 * mean(net.a{1}, 1);
    for i = 2 : n
        net.a{i} = sigm(repmat(net.b{i - 1}', m, 1) + net.a{i - 1} * net.W{i - 1}');
        if(net.dropoutFraction > 0 && i<n) 
            net.a{i} = net.a{i}.*(rand(size(net.a{i}))>net.dropoutFraction);
        end
        
        net.p{i} = 0.99 * net.p{i} + 0.01 * mean(net.a{i}, 1);
    end
    
    net.e = y - net.a{n};
    net.L = 1/2 * sum(sum(net.e .^ 2)) / m; 
end
