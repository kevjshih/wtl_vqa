function[] = vqa_extract_unique_words(vqa, out_fname)

%% extract all words from questions in answers
    annIds = vqa.getQuesIds();
    % load the annotations
    anns = vqa.loadQA(annIds);
    words_all = containers.Map();
    ids = cellfun(@double, cell(vqa.getQuesIds));
    for i = 1:length(anns)
        if mod(i, 100) == 0
            fprintf('%d/%d\n', i, length(anns));
        end
        words = get_unique_words(vqa.qqa{ids(i)}, anns{i});
        words_all = [words_all; words];        
    end
 
    save('words_all_map.mat', 'words_all', '-v7.3');
    word_cell = keys(words_all);
    %% now write it to some file
    fid = fopen(out_fname, 'w');
    for i = 1:length(word_cell)
        fprintf(fid, '%s\n', word_cell{i});
    end
    fclose(fid)
    
end

function[words_map] = get_unique_words(qs, anns)
    q_str = sanitize_string(char(qs{'question'}));
    answers = anns{'answers'};
    a_strs = cell(length(answers),1);

    for i = 1:length(answers)
        a_strs{i} = sanitize_string(char(answers{i}{'answer'}));
    end

    a_strs_all = cat(2, a_strs{:});    
    words = [q_str(:); a_strs_all(:)];
    words_map = containers.Map(words, ones(numel(words),1));
end
