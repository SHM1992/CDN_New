clear all;
clc;
% close all;
Reload_readear_Data = false; % True: get features again using the bounding boxes detected by the HOG+CSS+SVM
Init_udn = true;
Init_cnn = true;
AfterNMS = true;
train = true;%true:train all the para in the model
pretrain = true; %true: train all the deep networks
udn_pretrain = true;
nn_pretrain = true;
cdn_train = true; %true: train the deep neural network for collaboration
test = true;

if ~exist('Pathadd', 'var')
%    addpath .\gabor
    addpath ..\CNN
    addpath ..\toolbox-master\channels
    addpath ..\util
    addpath ..\tmptoolbox\matlab
    addpath ..\tmptoolbox\classify
    addpath ..\tmptoolbox
    addpath ..\tmptoolbox\images
    addpath ..\dbEval
    Pathadd = 1;
end;
wRatio=1.4;
hRatio=1.4;
Crop = [12+1 12+84; 5 5+28-1];
CropSize = Crop(:,2)-Crop(:,1)+1;

TrainCropImagepath=['../data/JDN data/CaltechTrain/' sprintf('w_%f_h_%f/',wRatio,hRatio)];


TrainCropImagesFName = [TrainCropImagepath 'CaltechTestAllimBoxesBeforeNmsRsz3'];
TrainCropLabelsFName = [TrainCropImagepath 'CaltechTestAllimBoxesBeforeNmsRszLabel3'];
TrainCropBoxesFName = [TrainCropImagepath 'CaltechTestAllimBoxesBeforeNmsRszBox3'];
ReaderDataFName = [TrainCropImagepath 'CNNDLTData3Color63_4.mat'];

dstCropImagepath=['../data/JDN data/CaltechTest/' sprintf('w_%f_h_%f/',wRatio,hRatio)];
TestCropImagesFName = [dstCropImagepath 'CaltechTestAllimBoxesBeforeNmsRsz2'];
TestCropLabelsFName = [dstCropImagepath 'CaltechTestAllimBoxesBeforeNmsRszLabel2'];
TestCropBoxesFName = [dstCropImagepath 'CaltechTestAllimBoxesBeforeNmsRszBox2'];

if Reload_readear_Data
    load(TrainCropImagesFName,'AllimBoxesBeforeNmsRsz');
    load(TrainCropLabelsFName, 'Labels');
    load(TrainCropBoxesFName, 'Allpartboxes');
    [train_x, train_y, Train_Boxes, Train_Frame] = GetData_datareader(AllimBoxesBeforeNmsRsz, Labels, Allpartboxes, 1, Crop); %
    
    
    load(TestCropImagesFName, 'AllimBoxesBeforeNmsRsz');
    load(TestCropLabelsFName, 'Labels');
    load(TestCropBoxesFName, 'Allpartboxes');
    [test_x, test_y, Test_Boxes, Test_Frame] = GetData_datareader(AllimBoxesBeforeNmsRsz, Labels, Allpartboxes, 0, Crop);
    save(ReaderDataFName, '-v7.3', 'train_x', 'train_y', 'test_x', 'test_y', 'Test_Boxes', 'Test_Frame', 'Train_Boxes');
    clear AllimBoxesBeforeNmsRsz Labels Allpartboxes;
else
    if ~exist('train_x', 'var') || ~exist('test_x', 'var')
        load(ReaderDataFName, 'train_x', 'train_y', 'test_x', 'test_y', 'Test_Boxes', 'Test_Frame', 'Train_Boxes');
    end
end


%% ex1
if train

    if pretrain
        %pre-train the component deep networks
        if udn_pretrain
             %clustering training samples
            fprintf('re-sampling...\n');
            [ClusterRes] = clustering(train_x,train_y,Train_Boxes,train_x);
            fprintf('done!\n');

            fprintf('pre-training UDN...\n');
            udn = cell(1,7);
            udn_best_index = [1,1];%select the trained model with minimum rs 
            %training UDNs
            fprintf('pre-training UDN No.1 to No. 6...\n');
            for i = 1:6
                if Init_udn || ~isfield(udn{i}, 'testrs')
                    udn{i}.layers = {
                        struct('type', 'i') %input layer
                        struct('type', 'c', 'outputmaps', 64, 'kernelsize', 9) %convolution layer
                        struct('type', 's', 'scale', 4) %sub sampling layer
                        struct('type', 'c', 'outputmaps', 20, 'kernelsize', 9) %convolution layer
                        };
                    load('./CNNModel_init.mat'); %initialization
                    udn{i} = udnsetup3(udn{i}, cnn_model, ClusterRes{2-mod(i,2),fix((i+1)/2)}.x, ClusterRes{2-mod(i,2),fix((i+1)/2)}.y, CropSize);
                end
                udn{i}.CropSize = CropSize;
                opts.alpha = 0.025;
                opts.batchsize = 71;
                opts.numepochs = 1;


                udn{i} = udntrain(udn{i},i, AfterNMS, ClusterRes{2-mod(i,2),fix((i+1)/2)}.x, ClusterRes{2-mod(i,2),fix((i+1)/2)}.y, opts, test_x, test_y, Test_Boxes, Test_Frame,ClusterRes{2-mod(i,2),fix((i+1)/2)}.boxes);
                fprintf('done!\n');
            end

            fprintf('training UDN...\n');
            if Init_udn || ~isfield(udn{7}, 'testrs')
                    udn{7}.layers = {
                        struct('type', 'i') %input layer
                        struct('type', 'c', 'outputmaps', 64, 'kernelsize', 9) %convolution layer
                        struct('type', 's', 'scale', 4) %sub sampling layer
                        struct('type', 'c', 'outputmaps', 20, 'kernelsize', 9) %convolution layer
                        };
                    load('./CNNModel_init.mat'); %initialization
                    udn{7} = udnsetup3(udn{7}, cnn_model, train_x, train_y, CropSize);
            end
                udn{7}.CropSize = CropSize;
                opts.alpha = 0.025;
                opts.batchsize = 71;
                opts.numepochs = 1;


                udn{7} = udntrain(udn{7}, 7, AfterNMS, train_x, train_y, opts, test_x, test_y, Test_Boxes, Test_Frame,Train_Boxes);
            fprintf('done!\n');
    % %         save('.\bestindex.mat','udn_best_index');
    % 
    %         fprintf('training CNN...\n');
    %         if Init_cnn || ~isfield(cnn, 'testrs')
    %             cnn.layers = {
    %                 struct('type', 'i') %input layer
    %                 struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) %convolution layer
    %                 struct('type', 's', 'scale', 2) %sub sampling layer
    %                 struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
    %                 struct('type', 's', 'scale', 2) %subsampling layer
    %                 };
    %         end
    %         opts.alpha = 0.1;
    %         opts.batchsize = 71;
    %         opts.numepochs = 1;
    % 
    %         cnn = cnnsetup(cnn, train_x, train_y);
    %         cnn = cnntrain(cnn, 1, train_x, train_y, opts);
    %         fprintf('done!\n');
        end

        if nn_pretrain
            %setup nn
            fprintf('pre-training nn...\n');

            x=zeros(50,14);
            y=zeros(2,50);
            for i = 1:30
                x(i,:) = [1,0,1,0,1,0,1,0,1,0,1,0,1,0];
                y(:,i) = [1;0];
            end
            for i = 31:50
                x(i,:) = [0,1,0,1,0,1,0,1,0,1,0,1,0,1];
                y(:,i) = [0;1];
            end
            nn = nnsetup([14,2]);
            nn.lambda = 1e-5;       %  L2 weight decay
            nn.alpha  = 1;       %  Learning rate
            nn_opts.numepochs =  1;   %  Number of full sweeps through data
            nn_opts.batchsize = 10;   %  Take a mean gradient step over this many samples
            nn = nntrain(nn, x, y', nn_opts);
            fprintf('\n');

        end
    end
    %     training CDN 
        if cdn_train
            fprintf('training CDN...\n');
            udn_opts.alpha = 0.025;
            udn_opts.batchsize = 71;
            udn_opts.numepochs = 1;
            cdntrain(train_x, train_y, udn_opts, udn_pretrain);        
            fprintf('done!\n');
        end    
end

%test 
if test
    %test UDNs
    fprintf('start testing...\n');
    out_input = zeros(size(test_x{1},3),14);
    out = cell(1,7);
%     load('.\bestindex.mat');
    for i = 1:7
         fprintf('loading udn_model...');
         fprintf('\n');        
         modelFname = ['.\New_UDNModel\UDN_Model_iter' num2str(i) '.1.mat'];
         if exist(modelFname,'file') == 0
            fprintf('error\n');
            return;
         else
            load(modelFname);
            out{i} = udntest(udn_model, test_x);
            out_input(:,2*i-1:2*i) = out{i}(:,:)';
         end 
%          [rs] = testCNNCaltechTest2(out{1}, Test_Boxes, Test_Frame);
%          fprintf('loading cnn_model...');
%          fprintf('\n');        
%          modelFname = ['.\New_CNNModel\CNN_Model_iter1.1.mat'];
%          if exist(modelFname,'file') == 0
%             fprintf('error\n');
%             return;
%          else
%             load(modelFname);
%             out{2} = cnntest(net, test_x);
%             out_input(:,3:4) = out{2}(:,:)';
%          end 
    end
    
    %test DNN
    modelFname = ['.\New_NNModel\NN_Model_iter1.mat'];
    if exist(modelFname,'file') == 0
        fprintf('error');
        return;
    else
        fprintf('loading nn_model\n');
        load(modelFname);
        outputs = nntest(nn,out_input,test_y');
        [rs] = testCNNCaltechTest2(outputs', Test_Boxes, Test_Frame);
        fprintf('the result is %f\n',rs);
    end
    fprintf('done!\n');
end



