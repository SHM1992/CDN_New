function [map] = Compute_rgbmap(images,labels,flag)
%compute the rgbmap of image I

map = cell(1,3);
dataNum = 1;
for i = 1:length(images)
    for j = 1:length(images{i})
        if abs(labels{i}(j)) > 0.1
            I = images{i}{j}.im;
            I = single(I);
            I1 = I(:,:,1);
            I2 = I(:,:,2);
            I3 = I(:,:,3);
            map{1}(:,:,dataNum) = (I1-mean(I1(:)))/(std(I1(:))+0.00001);
            map{2}(:,:,dataNum) = (I2-mean(I2(:)))/(std(I2(:))+0.00001);
            map{3}(:,:,dataNum) = (I3-mean(I3(:)))/(std(I3(:))+0.00001);
            dataNum = dataNum + 1;
            if flag == 1 && labels{i}(j)>0
                    I = images{i}{j}.im(:,end:-1:1,:);
                    I = single(I);
                    I1 = I(:,:,1);
                    I2 = I(:,:,2);
                    I3 = I(:,:,3);
                    map{1}(:,:,dataNum) = (I1-mean(I1(:)))/(std(I1(:))+0.00001);
                    map{2}(:,:,dataNum) = (I2-mean(I2(:)))/(std(I2(:))+0.00001);
                    map{3}(:,:,dataNum) = (I3-mean(I3(:)))/(std(I3(:))+0.00001);
                    dataNum = dataNum + 1;
            end
        end
    end
end

end

