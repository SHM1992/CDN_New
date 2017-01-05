function [out] = udntest(net, x)
    %  feedforward
    net = udnff(net, x);
    out = net.o;
%     [~, h] = max(net.o);
%     [~, a] = max(y);
%     bad = find(h ~= a);
% 
%     er = numel(bad) / size(y, 2);
end
