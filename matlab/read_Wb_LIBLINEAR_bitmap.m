function [W, b, bias] = read_Wb_LIBLINEAR_bitmap(model_fileName)


%  * FORMAT MODEL FILE:
%  * <solver type>: int
%  * <nr_class>: int
%  * <labels>: 'nr_class'-length vector of int
%  * <nr_feature>: int
%  * <bias>: double
%  * <Wb>: 'nr_feature*nr_class' float matrix in row-order

fin = fopen(model_fileName, 'r');

solver_type = fread(fin, 1, 'int');
nr_class = fread(fin, 1, 'int');
labels = fread(fin, nr_class, 'int');
nr_feature = fread(fin, 1, 'int');
bias = fread(fin, 1, 'double');
Wb = fread(fin, [nr_class nr_feature], '*float');

fclose(fin);

% N.B.: re-order Wb!!
[junk idx_wb_ordered] = sort(labels); 
Wb = Wb(idx_wb_ordered, :);

% split W into [W,b] (if necessary)
if bias >= 0
    b = Wb(:,nr_feature);
    W = Wb(:,1:nr_feature-1);
else
    b = inf*ones(nr_class,1);
end

end