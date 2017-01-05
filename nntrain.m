function nn = nntrain(nn, x, y, opts)
    assert(isfloat(x), 'x must be a float');
    m = size(x, 1);
    
    batchsize = opts.batchsize;
    numepochs = opts.numepochs;

    numbatches = m / batchsize;

    assert(rem(numbatches, 1) == 0, 'numbatches not integer');

    nn.rL = [];
    n = 1;
    min_er = 1;
    index = 1;
    
    for i = 1 : numepochs
        tic;

        kk = randperm(m);
        %output weights in every epoch
%         num = 1;
%         weights{i} = zeros(numbatches,14);
        for l = 1 : numbatches
            batch_x = x(kk((l - 1) * batchsize + 1 : l * batchsize), :);
            %add noise to input (for use in denoising autoencoder)
            if(nn.inputZeroMaskedFraction ~= 0)
                batch_x = batch_x.*(rand(size(batch_x))>nn.inputZeroMaskedFraction);
            end
            
            batch_y = y(kk((l - 1) * batchsize + 1 : l * batchsize), :);
            
%             weights{i}(num,:) = reshape(nn.W{1},1,14);num = num + 1;
            nn = nnff(nn, batch_x, batch_y);
            nn = nnbp(nn);            
            nn = nnapplygrads(nn);

            if n == 1
                nn.rL(n) = nn.L;
            end

            nn.rL(n + 1) = 0.99 * nn.rL(n) + 0.01 * nn.L;
            n = n + 1;
        end

        t = toc;
%         save('.\weights','weights');
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs) '. Took ' num2str(t) ' seconds' '. Mean squared error is ' num2str(nn.rL(end))]);
        if nn.rL(end)<min_er
            min_er = nn.rL(end);
            index = i;
        end
        ModelFname = ['.\New_NNModel\NN_Model_iter' num2str(i) '.mat'];
        save(ModelFname, 'nn');
    end
end

