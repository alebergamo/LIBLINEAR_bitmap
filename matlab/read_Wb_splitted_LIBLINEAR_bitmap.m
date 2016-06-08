function [W, b] = read_Wb_splitted_LIBLINEAR_bitmap(model_fileName, delete_files)
% NOTE: There is supposed to be files <model_fileName>_cl0 etc...

if nargin < 2
    delete_files = 0;
end

model_fileName_str = [model_fileName '_cl%d'];

idx_cl = 0;
[temp_w, temp_b, nr_class] = read_Wb_splitted_LIBLINEAR_support(sprintf(model_fileName_str,idx_cl));
W = ones(nr_class, length(temp_w),'single');
b = ones(nr_class, length(temp_b),'single');
W(1,:) = temp_w;
b(1,:) = temp_b;
for idx_cl=1:(nr_class-1)
    progress_bar(idx_cl, nr_class-1, 5);
    
    [temp_w, temp_b, nr_class, labels] = read_Wb_splitted_LIBLINEAR_support(sprintf(model_fileName_str,idx_cl));
    W(idx_cl+1,:) = temp_w;
    b(idx_cl+1,:) = temp_b;
end

% N.B.: re-order Wb!!
[junk idx_wb_ordered] = sort(labels); 
W = W(idx_wb_ordered, :);
b = b(idx_wb_ordered);

% if requested, let's delete the files
if delete_files
    for idx_cl=0:(nr_class-1)
        unix_command(['rm -f ' sprintf(model_fileName_str,idx_cl)]);
    end
end

end



function [W, b, nr_class, labels] = read_Wb_splitted_LIBLINEAR_support(model_fileName)

fin = fopen(model_fileName, 'r');

solver_type = fread(fin, 1, 'int');
nr_class = fread(fin, 1, 'int');
labels = fread(fin, nr_class, 'int');
nr_feature = fread(fin, 1, 'int');
bias = fread(fin, 1, 'double');
Wb = fread(fin, [1 nr_feature], '*float');

fclose(fin);

if bias >= 0
    b = Wb(nr_feature);
    W = Wb(1:nr_feature-1);
else
    b = inf*ones(nr_class,1);
end

end