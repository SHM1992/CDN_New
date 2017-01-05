function [map] = Compute_yuvmap(images,labels,flag)
%compute the yuvmap of image I

map = cell(1,3);
dataNum = 1;
for i = 1:length(images)
    for j = 1:length(images{i})
        if flag == 1 && labels{i}(j)==0
            continue;
        else
            I = images{i}{j}.im;
            YUVI = RGB2YUV(I);
            YUVI = single(YUVI);
            map{1}(:,:,dataNum) = YUVI(:,:,1);
            map{2}(:,:,dataNum) = YUVI(:,:,2);
            map{3}(:,:,dataNum) = YUVI(:,:,3);
            dataNum = dataNum + 1;
            if flag == 1
                if labels{i}(j)>0
                    I = images{i}{j}.im(:,end:-1:1,:);
                    YUVI = RGB2YUV(I);
                    YUVI = single(YUVI);                   
                    map{1}(:,:,dataNum) = YUVI(:,:,1);
                    map{2}(:,:,dataNum) = YUVI(:,:,2);
                    map{3}(:,:,dataNum) = YUVI(:,:,3);
                    dataNum = dataNum + 1;
                end
            end
        end
    end
end

end