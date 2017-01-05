function [ HOGmap ] = Compute_yuvHOGmap( yuvImages )
%COMPUTE_YUVHOGMAP compute the HOGmap of images

HOGmap = cell(1,3);
for k = 1:3
    for l = 1:size(yuvImages{k},3)
        I = yuvImages{k}(:,:,l);
        I = sqrt(I);      %伽马校正
        I = real(I);
        [m,n]=size(I);

        %求边缘
        fy=[-1 0 1];        %定义竖直模板
        fx=fy';             %定义水平模板
        Iy=imfilter(I,fy,'replicate');    %竖直边缘
        Ix=imfilter(I,fx,'replicate');    %水平边缘
        Ied=sqrt(Ix.^2+Iy.^2);              %边缘强度
        Iphase=Iy./Ix;                      %边缘斜率，有些为inf,-inf,nan，其中nan需要再处理一下

        %求cell
        cellsize=4;            %the best setting in PD is cellsize=6,blocksize=3,bin=9
                               %for transforming 108*36 to 84*28,HOG channel needs 9*3 blocks
                               %each block has 81 dimensions(3*3cells*9bins))
                               %so computing region is 12*12                         
        orient=9;               %方向直方图的方向个数
        jiao=360/orient;        %每个方向包含的角度数
        Cell=cell(1,1);         %所有的角度直方图,cell是可以动态增加的，所以先设了一个
        ii = 1;                  %ii is the cell row index
        jj = 1;                  %jj is the cell col index
        for i = 1:cellsize:m          %imgi is the row index of start pixel             
            for j = 1:cellsize:n      %imgj is the col index of start pixel           
                tmpx = Ix(i:i+cellsize-1,j:j+cellsize-1);
                tmped = Ied(i:i+cellsize-1,j:j+cellsize-1);
                tmped = tmped/sum(sum(tmped));        %局部边缘强度归一化
                tmpphase = Iphase(i:i+cellsize-1,j:j+cellsize-1);
                Hist = zeros(1,orient);               %当前cellsize*cellsize像素块统计角度直方图,就是cell
                for p = 1:cellsize
                    for q = 1:cellsize
                        if isnan(tmpphase(p,q)) == 1  %0/0会得到nan，如果像素是nan，重设为0
                            tmpphase(p,q) = 0;
                        end
                        ang = atan(tmpphase(p,q));    %atan求的是[-90 90]度之间
                        ang = mod(ang*180/pi,360);    %全部变正，-90变270
                        if tmpx(p,q)<0              %根据x方向确定真正的角度
                            if ang<90               %如果是第一象限
                                ang = ang+180;        %移到第三象限
                            end
                            if ang>270              %如果是第四象限
                                ang = ang-180;        %移到第二象限
                            end
                        end
                        ang = ang + 0.0000001;          %防止ang为0
                        Hist(ceil(ang/jiao)) = Hist(ceil(ang/jiao)) + tmped(p,q);   %ceil向上取整，使用边缘强度加权
                    end
                end
                Hist = Hist/sum(Hist);    %方向直方图归一化
                Cell{ii,jj} = Hist;       %放入Cell中
                jj = jj + 1;                
            end
            jj = 1;
            ii = ii + 1;                    
        end

        %compute HOG channel 
        [m,n]=size(Cell);
        for HMi = 1:m           
            for HMj = 1:n           
                HOGmap{k}((HMi-1)*3+1:HMi*3,(HMj-1)*3+1:HMj*3,l) = reshape(Cell{HMi,HMj},3,3);
            end
        end
        for HMii = size(HOGmap{k}(:,:,l),1):84
            for HMjj = size(HOGmap{k}(:,:,l),2):28
            HOGmap{k}(HMii,HMjj,l) = 0;     %zero padding
            end
        end
    end
end

end

