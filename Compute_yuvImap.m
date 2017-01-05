function [ Imap ] = Compute_yuvImap( yuvImages )
%COMPUTE_YUVIMAP compute integral maps of yuv maps

Imap = cell(1,3);
for i = 1:3
    for j = 1:size(yuvImages{i},3)
        I = yuvImages{i}(:,:,j);
        Imap{i}(1,1,j) = I(1,1);
        for k = 1:size(I,1)
            for l = 1:size(I,2)
                if k==1 && l==1
                    continue;
                end
                if k==1
                    Imap{i}(k,l,j) = Imap{i}(k,l-1,j)+I(k,l);
                end
                if l==1
                    Imap{i}(k,l,j) = Imap{i}(k-1,l,j)+I(k,l);
                end
                if k>1 && l>1
                    Imap{i}(k,l,j) = Imap{i}(k,l-1,j)+Imap{i}(k-1,l,j)-Imap{i}(k-1,l-1,j)+I(k,l);
                end
            end
        end
    end
end

end

