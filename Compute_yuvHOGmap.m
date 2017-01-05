function [ HOGmap ] = Compute_yuvHOGmap( yuvImages )
%COMPUTE_YUVHOGMAP compute the HOGmap of images

HOGmap = cell(1,3);
for k = 1:3
    for l = 1:size(yuvImages{k},3)
        I = yuvImages{k}(:,:,l);
        I = sqrt(I);      %٤��У��
        I = real(I);
        [m,n]=size(I);

        %���Ե
        fy=[-1 0 1];        %������ֱģ��
        fx=fy';             %����ˮƽģ��
        Iy=imfilter(I,fy,'replicate');    %��ֱ��Ե
        Ix=imfilter(I,fx,'replicate');    %ˮƽ��Ե
        Ied=sqrt(Ix.^2+Iy.^2);              %��Եǿ��
        Iphase=Iy./Ix;                      %��Եб�ʣ���ЩΪinf,-inf,nan������nan��Ҫ�ٴ���һ��

        %��cell
        cellsize=4;            %the best setting in PD is cellsize=6,blocksize=3,bin=9
                               %for transforming 108*36 to 84*28,HOG channel needs 9*3 blocks
                               %each block has 81 dimensions(3*3cells*9bins))
                               %so computing region is 12*12                         
        orient=9;               %����ֱ��ͼ�ķ������
        jiao=360/orient;        %ÿ����������ĽǶ���
        Cell=cell(1,1);         %���еĽǶ�ֱ��ͼ,cell�ǿ��Զ�̬���ӵģ�����������һ��
        ii = 1;                  %ii is the cell row index
        jj = 1;                  %jj is the cell col index
        for i = 1:cellsize:m          %imgi is the row index of start pixel             
            for j = 1:cellsize:n      %imgj is the col index of start pixel           
                tmpx = Ix(i:i+cellsize-1,j:j+cellsize-1);
                tmped = Ied(i:i+cellsize-1,j:j+cellsize-1);
                tmped = tmped/sum(sum(tmped));        %�ֲ���Եǿ�ȹ�һ��
                tmpphase = Iphase(i:i+cellsize-1,j:j+cellsize-1);
                Hist = zeros(1,orient);               %��ǰcellsize*cellsize���ؿ�ͳ�ƽǶ�ֱ��ͼ,����cell
                for p = 1:cellsize
                    for q = 1:cellsize
                        if isnan(tmpphase(p,q)) == 1  %0/0��õ�nan�����������nan������Ϊ0
                            tmpphase(p,q) = 0;
                        end
                        ang = atan(tmpphase(p,q));    %atan�����[-90 90]��֮��
                        ang = mod(ang*180/pi,360);    %ȫ��������-90��270
                        if tmpx(p,q)<0              %����x����ȷ�������ĽǶ�
                            if ang<90               %����ǵ�һ����
                                ang = ang+180;        %�Ƶ���������
                            end
                            if ang>270              %����ǵ�������
                                ang = ang-180;        %�Ƶ��ڶ�����
                            end
                        end
                        ang = ang + 0.0000001;          %��ֹangΪ0
                        Hist(ceil(ang/jiao)) = Hist(ceil(ang/jiao)) + tmped(p,q);   %ceil����ȡ����ʹ�ñ�Եǿ�ȼ�Ȩ
                    end
                end
                Hist = Hist/sum(Hist);    %����ֱ��ͼ��һ��
                Cell{ii,jj} = Hist;       %����Cell��
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

