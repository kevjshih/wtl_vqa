function[] = vqa_extract_unique_words_test(testqs, out_fname)

%% extract all words from questions in answers
    num_qs = length(testqs{'questions'});
    % load the annotations
    words_all = containers.Map();
    for i = 1:num_qs
        if mod(i, 100) == 0
            fprintf('%d/%d\n', i, num_qs);
        end
        words = get_unique_words(testqs{'questions'}{i});
        words_all = [words_all; words];        
    end
    
    %    save('words_all_map_testdev.mat', 'words_all', '-v7.3');
    save('words_all_map_test.mat', 'words_all', '-v7.3');
    word_cell = keys(words_all);
    %% now write it to some file
    fid = fopen(out_fname, 'w');
    for i = 1:length(word_cell)
        fprintf(fid, '%s\n', word_cell{i});
    end
    fclose(fid)
    
end


function[words_map] = get_unique_words(qs)
    q_str = sanitize_string(char(qs{'question'}));
    answers = qs{'multiple_choices'};
    a_strs = cell(length(answers),1);

    for i = 1:length(answers)
        a_strs{i} = sanitize_string(char(answers{i}));
    end
    a_strs_all = cat(2, a_strs{:});    
    words = [q_str(:); a_strs_all(:)];
    words_map = containers.Map(words, ones(numel(words),1));
end
