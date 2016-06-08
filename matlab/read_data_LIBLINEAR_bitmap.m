function [fstar_bin tr_label] = read_data_LIBLINEAR_bitmap(data_fileName)
%  
% FORMAT:
% <uint> :  number of dimensions
% <uint> :  number of examples
% <int>*n_examples  : labels
% <data> We store the bits in colum-order (load uchar)
%

fid = fopen(data_fileName, 'r');
if fid < 0
    error(['Error during opening the file ' data_fileName]);
end

% read the header
rows = double(fread(fid, 1, 'uint'));
cols = double(fread(fid, 1, 'uint'));
tr_label = double(fread(fid, cols, 'int'));

% read the data
fstar_bin = fread(fid, rows*cols/8, '*uint8');

fclose(fid);

end