
%load data
if ~exist('Pathadd', 'var')
%    addpath .\gabor
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

TrainCropImagepath=['../data/JDN data/CaltechTrain/' sprintf('w_%f_h_%f/',wRatio,hRatio)];

ReaderDataFName = [TrainCropImagepath 'CNNDLTData3Color63_4.mat'];

if ~exist('train_x', 'var') || ~exist('test_x', 'var')
        load(ReaderDataFName, 'train_x', 'train_y', 'test_x', 'test_y', 'Test_Boxes', 'Test_Frame', 'Train_Boxes');
end

%clustering
fprintf('clustering...\n');
[clusterRes1,clusterRes2,clusterRes3] = clustering(train_x,train_y,Train_Boxes,2,2,2);

%save different clusters in different folders
fprintf('writing images')
for i = 1:size(clusterRes1)
    for j = 1:size(clusterRes1{i}.x{3},3)
        I(:,:,1) = clusterRes1{i}.x{1}(:,:,j);
        I(:,:,2) = clusterRes1{i}.x{2}(:,:,j);
        I(:,:,3) = clusterRes1{i}.x{3}(:,:,j);
        imwrite(I,['.\Cluster',num2str(i),'onChannel1\image',num2str(j)],'jpg');
    end
end
for i = 1:size(clusterRes2)
    for j = 1:size(clusterRes2{i}.x{3},3)
        I(:,:,1) = clusterRes2{i}.x{1}(:,:,j);
        I(:,:,2) = clusterRes2{i}.x{2}(:,:,j);
        I(:,:,3) = clusterRes2{i}.x{3}(:,:,j);
        imwrite(I,['.\Cluster',num2str(i),'onChannel2\image',num2str(j)],'jpg');
    end
end
for i = 1:size(clusterRes3)
    for j = 1:size(clusterRes3{i}.x{3},3)
        I(:,:,1) = clusterRes3{i}.x{1}(:,:,j);
        I(:,:,2) = clusterRes3{i}.x{2}(:,:,j);
        I(:,:,3) = clusterRes3{i}.x{3}(:,:,j);
        imwrite(I,['.\Cluster',num2str(i),'onChannel3\image',num2str(j)],'jpg');
    end
end