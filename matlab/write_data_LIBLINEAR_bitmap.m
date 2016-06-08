function write_data_LIBLINEAR_bitmap(fstar_bin, tr_label, outputFile, overwrite)
% fstar must be a column vector uint8. The # rows of the bitmap MUST be a multiple of 8
%  
% FORMAT:
% <uint> :  number of dimensions
% <uint> :  number of examples
% <int>*n_examples  : labels
% <data> We store the bits in colum-order (load uchar)
%

if size(fstar_bin,2) > 1
   error('fstar_bin must be a column vector'); 
end
if ~isa(fstar_bin, 'uint8')
    error('The input data must be uint8');
end
if mod(length(fstar_bin)*8, length(tr_label)) ~= 0
   error('The dimensionality of the data must be a multiple of 8'); 
end

rows = length(fstar_bin)*8/length(tr_label);
cols = length(tr_label);

if exist(outputFile,'file') && ~overwrite
   return; 
end

fid = fopen(outputFile, 'w');
if fid < 0
    error(['Error during opening the file ' outputFile]);
end

% write the header
fwrite(fid, rows, 'uint');
fwrite(fid, cols, 'uint');
fwrite(fid, tr_label, 'int');

% write the data
fwrite(fid, fstar_bin, 'uint8');

% M = reshape(M, rows*cols, 1);
% M_bin = zeros(rows*cols/8, 1, 'uint8');
% M_bin = bitset(M_bin, 8, M(1:8:end));
% M_bin = bitset(M_bin, 7, M(2:8:end));
% M_bin = bitset(M_bin, 6, M(3:8:end));
% M_bin = bitset(M_bin, 5, M(4:8:end));
% M_bin = bitset(M_bin, 4, M(5:8:end));
% M_bin = bitset(M_bin, 3, M(6:8:end));
% M_bin = bitset(M_bin, 2, M(7:8:end));
% M_bin = bitset(M_bin, 1, M(8:8:end));
% fwrite(fid, M_bin, 'uint8');

fclose(fid);

end