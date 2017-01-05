function cdntrain(x, y, udn_opts, udn_pretrain)
%CDNSETUP setup cdn model
%   --cdn_model: including parameters settring of cdn model
%   --x: training samples
%   --y: training labels
%   --opts: training settrings:
%           --cluster_algs: parameters of clustering for preparing subsets
%           --CropSize: parameter using in UDN setup

if ~udn_pretrain
    Init_udn = true;
    Crop = [12+1 12+84; 5 5+28-1];
    CropSize = Crop(:,2)-Crop(:,1)+1;
    udn = cell(1,7);
    for i = 1:7
        if Init_udn || ~isfield(udn{i}, 'testrs')
            udn{i}.layers = {
                struct('type', 'i') %input layer
                struct('type', 'c', 'outputmaps', 64, 'kernelsize', 9) %convolution layer
                struct('type', 's', 'scale', 4) %sub sampling layer
                struct('type', 'c', 'outputmaps', 20, 'kernelsize', 9) %convolution layer
                };
            load('./CNNModel_init.mat'); %initialization
            udn{i} = udnsetup3(udn{i}, cnn_model, x, y, CropSize);
        end
    end
end
m = size(x{1},3);
batchsize = udn_opts.batchsize;
numbatches = m / batchsize;
assert(rem(numbatches,1)==0, 'numbatches not integer');

tic;
kk = randperm(m);
for i = 1:numbatches
    fprintf(['training CDN numbatches ' num2str(i) '...\n']);
    batch_x{1} = x{1}(:, :, kk( (i-1) * batchsize + 1 : i * batchsize));
    batch_x{2} = x{2}(:, :, kk( (i-1) * batchsize + 1 : i * batchsize));
    batch_x{3} = x{3}(:, :, kk( (i-1) * batchsize + 1 : i * batchsize));
    batch_y = y(:, kk( (i-1) * batchsize + 1 : i * batchsize));
    
    fprintf('  testing multiple deep networks using training samples...\n');
    out_input = zeros(size(batch_x{1},3),14);
    out = cell(1,7);
    for j = 1:7
        fprintf('  loading udn_model...');
        fprintf('\n');      
        modelFname = ['.\New_UDNModel\UDN_Model_iter' num2str(j) '.1.mat'];
        if exist(modelFname,'file') == 0
            fprintf('  error\n');
            return;
        else
            load(modelFname);
            fprintf('  testing UDN...\n');
            out{j} = udntest(udn_model, batch_x);%In fact, it does udnff
            out_input(:,2*j-1:2*j) = out{j}(:,:)';
            fprintf('  done!\n');
        end 
        
%         fprintf('  loading cnn_model...');
%         fprintf('\n');      
%         modelFname = ['.\New_CNNModel\CNN_Model_iter1.1.mat'];
%         if exist(modelFname,'file') == 0
%             fprintf('  error\n');
%             return;
%         else
%             load(modelFname);
%             fprintf('  testing CNN...\n');
%             out{2} = cnntest(net, batch_x);
%             out_input(:,3:4) = out{2}(:,:)';
%             fprintf('  done!\n');
%         end 
    end
    fprintf('  done!\n');
    
    fprintf('  training nn...\n');
    modelFname = ['.\New_NNModel\NN_Model_iter1.mat'];
    if exist(modelFname,'file') == 0
        fprintf('error');
        return;
    else
        fprintf('loading nn_model\n');
        load(modelFname);
        fprintf('fine-tuning nn...\n');
        %add noise to input (for use in denoising autoencoder)
        if(nn.inputZeroMaskedFraction ~= 0)
            out_input = out_input.*(rand(size(out_input))>nn.inputZeroMaskedFraction);
        end
        nn = nnff(nn, out_input, batch_y');
        nn = nnbp(nn);
        nn = nnapplygrads(nn);
        save('.\New_NNModel\NN_Model_iter1.mat', 'nn');
    end
    fprintf('  done!\n');
    
    fprintf('  fine-tuning UDN models...\n');
    for j = 1:7
        fprintf('  loading udn_model...');
        fprintf('\n');      
        modelFname = ['.\New_UDNModel\UDN_Model_iter' num2str(j) '.1.mat'];
        if exist(modelFname,'file') == 0
            fprintf('  error\n');
            return;
        else
            load(modelFname);
            udn_model.batchl = i;
            udn_model.batchMaxl = 10;
%             num = 1;
%             wrong_index = zeros(1,1);
%             for k = 1:size(out_input,1)
%                 if (out_input(k,2*j-1) >= out_input(k,2*j) && batch_y(1,k) < batch_y(2,k)) || (out_input(k,2*j-1) < out_input(k,2*j) && batch_y(1,k) >= batch_y(2,k))
%                     wrong_index(num) = k;
%                     num = num + 1;
%                 end
%             end
%             if num > 2
%                 for k = 1:length(batch_x)
%                     batch_wrong_x{k} = batch_x{k}(:,:,wrong_index);
%                 end
                udn_model = udnff(udn_model, batch_x);
                udn_model = udnbp(udn_model, batch_y, 1, (nn.d{1}(:,2*j-1:2*j))');
                udn_model = udnapplygrads(udn_model, udn_opts);
%             end
            save(modelFname, 'udn_model');
        end 
        
%         fprintf('  loading cnn_model...');
%         fprintf('\n');      
%         modelFname = ['.\New_CNNModel\CNN_Model_iter1.1.mat'];
%         if exist(modelFname,'file') == 0
%             fprintf('  error\n');
%             return;
%         else
%             load(modelFname);
%             net.batchl = i;
%             net.batchMaxl = 10;
%             net = cnnff(net, batch_x);
%             net = cnnbp(net, batch_y, 1, (nn.d{1}(:,3:4))');
%             net = cnnapplygrads(net, cnn_opts);
%             save(modelFname, 'net');
%         end 
    end
    fprintf('  done!\n');
    
end

t = toc;
fprintf('training time is %d\n' ,t);



end

