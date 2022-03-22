% Concatenate data features, files that share the same name will be concatented. 
% each file contains fi,i=1,..,4.
% f3 contains imagery number, use real part only;
% f4 has different dimension and cannot be concatented with f1-f3. 
% Yongwei Wang
% May 27, 2021

close all;
clear; clc

mat_files = dir('*.mat'); 
save_dir_base = '..\\Feature_data_processed\\'; 

for k = 1: length(mat_files)
    file_name = mat_files(k).name;
    cur_name = split(file_name, '~'); cur_name = char(cur_name(1));
    if k == 1
        pre_name = cur_name; 
        pre_data = double.empty();
    end
    
    cur_data_tmp = load(file_name);
    cur_data = cat(3, cur_data_tmp.f1, cur_data_tmp.f2, real(cur_data_tmp.f3)); 
    
    if string(cur_name) == string(pre_name)
        data_comb = cat(1, pre_data, cur_data);  
    else 
        data_comb = cur_data;
    end
    
    pre_data = data_comb;
    pre_name = cur_name; 
    
    % save data_comb
    disp(size(data_comb));
    save_file_path = [save_dir_base, cur_name, '.mat'];
    save(save_file_path, 'data_comb');
        
%     folder_name_tmp = split(file_name, '_'); 
%     folder_name = char(folder_name_tmp(2)); 
      
end

