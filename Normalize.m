function [ new_data ] = Normalize( data )
%πÈ“ªªØ

new_data = cell(1,3);
for i = 1:length(data)
    for j = 1:size(data{i},3)
        I = data{i}(:,:,j);
        new_data{i}(:,:,j) = (I-mean(I(:)))/(std(I(:))+0.00001);
        new_data{i}(:,:,j) = single(new_data{i}(:,:,j));
        new_data{i}(:,:,j) = squeeze(new_data{i}(:,:,j));
    end
end

end

