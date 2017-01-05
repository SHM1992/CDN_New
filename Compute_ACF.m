function [map] = Compute_ACF(images,labels,flag)
%compute the yuvmap of image I

map = cell(1,10);
dataNum = 1;
pChns = chnsCompute();
for i = 1:length(images)
    for j = 1:length(images{i})
        if abs(labels{i}(j)) > 0.1
            I = images{i}{j}.im;
            chns = chnsCompute(I,pChns);
%             switch type
%                 case '123'
%                     map{1}(:,:,dataNum) = chns.data{1}(:,:,1);
%                     map{2}(:,:,dataNum) = chns.data{1}(:,:,2);
%                     map{3}(:,:,dataNum) = chns.data{1}(:,:,3);
%                 case '567'
%                     map{1}(:,:,dataNum) = chns.data{3}(:,:,1);
%                     map{2}(:,:,dataNum) = chns.data{3}(:,:,2);
%                     map{3}(:,:,dataNum) = chns.data{3}(:,:,3);
%             end
            for k = 1:3
                map{k}(:,:,dataNum) = chns.data{1}(:,:,k);
            end
            map{4}(:,:,dataNum) = chns.data{2}(:,:);
            for k = 5:10
                map{k}(:,:,dataNum) = chns.data{3}(:,:,k-4);
            end
            dataNum = dataNum + 1;
            if flag == 1 && labels{i}(j)>0
                I = images{i}{j}.im(:,end:-1:1,:);  
                chns = chnsCompute(I,pChns);
%                 switch type
%                     case '123'
%                         map{1}(:,:,dataNum) = chns.data{1}(:,:,1);
%                         map{2}(:,:,dataNum) = chns.data{1}(:,:,2);
%                         map{3}(:,:,dataNum) = chns.data{1}(:,:,3);
%                     case '567'
%                         map{1}(:,:,dataNum) = chns.data{3}(:,:,1);
%                         map{2}(:,:,dataNum) = chns.data{3}(:,:,2);
%                         map{3}(:,:,dataNum) = chns.data{3}(:,:,3);
%                 end
                for k = 1:3
                    map{k}(:,:,dataNum) = chns.data{1}(:,:,k);
                end
                map{4}(:,:,dataNum) = chns.data{2}(:,:);
                for k = 5:10
                    map{k}(:,:,dataNum) = chns.data{3}(:,:,k-4);
                end
                dataNum = dataNum + 1;
            end
        end
    end
end
end