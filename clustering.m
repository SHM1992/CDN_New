function [Res] = clustering(x,y,boxes,features)
%this function is used to cluster the input training samples into several classes 
%for training different JDNs
%Input:
%       x:3-channels training samples, size is 3 cells, each including 84*28*64468
%       y:labels of training samples, size is 2*64468
%       boxes:locations and scores of each bounding boxes for training samples, size is 64468*5
%       features:ACF features,clustering criterion for training samples, size is 10 cells, each including 27*9*64468
%Output:
%       Res: clustering results of training samples,size is 2*10 cells,each
%       colum stores 2 clusters classified by each criterion(channel)

Res = cell(2,3);
for k = 1:3
    %k-means:
    num = size(features{k},3);
    if num < 2
        fprintf('error');
        return;
    end

    %initial value of centroid
    p = randperm(num);
    for i = 1:2
        c(:,:,i) = features{k}(:,:,p(i));
    end

    temp = zeros(num,1);%temp is used to be category vector for each iteration

    %clustering
    while 1
        d = DistMatrix(features{k},c);
        min_d = min(d,[],2);
        class = zeros(num,1);
        for i = 1:num
            class(i) = find(d(i,:) == min_d(i),1);
        end
        if class == temp
            break;
        else
            temp = class;
        end
        for i = 1:2
            if isempty(find(class == i))
                continue;
            else
                c(:,:,i) = mean(features{k}(:,:,find(class == i)),3);
            end
        end
    end


    inds1 = find(class == 1);
    inds2 = find(class == 2);
    if isempty(inds1)
        continue;
    end
    n1 = 1;n2 = 1;
    for q = 1:fix(length(inds1)/2)
        Res{1,k}.x{1}(:,:,n1) = x{1}(:,:,inds1(q));
        Res{1,k}.x{2}(:,:,n1) = x{2}(:,:,inds1(q));
        Res{1,k}.x{3}(:,:,n1) = x{3}(:,:,inds1(q));
        Res{1,k}.y(:,n1) = y(:,inds1(q));
        Res{1,k}.boxes(n1,:) = boxes(inds1(q),:);
        n1 = n1 + 1;
    end
    for q = 1:fix(length(inds2)/2)
        Res{1,k}.x{1}(:,:,n1) = x{1}(:,:,inds2(q));
        Res{1,k}.x{2}(:,:,n1) = x{2}(:,:,inds2(q));
        Res{1,k}.x{3}(:,:,n1) = x{3}(:,:,inds2(q));
        Res{1,k}.y(:,n1) = y(:,inds2(q));
        Res{1,k}.boxes(n1,:) = boxes(inds2(q),:);  
        n1 = n1 + 1;
    end
    Res{1,k}.num = fix(length(inds1)/2)+fix(length(inds2)/2);
    for q = fix(length(inds1)/2)+1:length(inds1)
        Res{2,k}.x{1}(:,:,n2) = x{1}(:,:,inds1(q));
        Res{2,k}.x{2}(:,:,n2) = x{2}(:,:,inds1(q));
        Res{2,k}.x{3}(:,:,n2) = x{3}(:,:,inds1(q));
        Res{2,k}.y(:,n2) = y(:,inds1(q));
        Res{2,k}.boxes(n2,:) = boxes(inds1(q),:);
        n2 = n2 + 1;
    end
    for q = fix(length(inds2)/2)+1:length(inds2)
        Res{2,k}.x{1}(:,:,n2) = x{1}(:,:,inds2(q));
        Res{2,k}.x{2}(:,:,n2) = x{2}(:,:,inds2(q));
        Res{2,k}.x{3}(:,:,n2) = x{3}(:,:,inds2(q));
        Res{2,k}.y(:,n2) = y(:,inds2(q));
        Res{2,k}.boxes(n2,:) = boxes(inds2(q),:);
        n2 = n2 + 1;
    end
    Res{2,k}.num = length(inds1)+length(inds2)-(fix(length(inds1)/2)+fix(length(inds2)/2));
end



function d = DistMatrix(x,c)
%this function is used to compute the distance between data features and centroid matrix

for i = 1:size(x,3)
    for j = 1:size(c,3)
        d(i,j) = norm(x(:,:,i)-c(:,:,j),2);
    end
end
