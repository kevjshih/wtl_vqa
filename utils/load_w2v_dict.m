function[dict, vec_matrix] = load_w2v_dict(word_file, vec_file, to_rows)
% if to_rows == 1
% dict is a map from word to rew in vec_file and vec_matrix is a matrix with columns of w2v vectors
% indexing done by w2v_vector = vec_matrix(:,dict('query_word'))
% otherwise it's a map to the vector from vec_file and vec_matrix is []
    if nargin < 3
        to_rows = 0;
    end
    if to_rows == 1
        vec_matrix = {}
    else        
        vec_matrix = [];
    end
    fid = fopen(word_file, 'r')
    fvec = fopen(vec_file)
    dict = containers.Map;
    cnt = 1;
    tline = fgets(fid);
    while ischar(tline) 
        
        vec = fread(fvec, 300, 'single');
        if to_rows == 1
            dict(strtrim(tline)) = cnt;
            vec_matrix{end+1} = vec;
        else
            dict(strtrim(tline)) = vec;
        end
        cnt = cnt+1;
        tline = fgets(fid);
    end
    if to_rows == 1
        vec_matrix = cat(2, vec_matrix{:});
    end
    fclose(fid)
    fclose(fvec)
end