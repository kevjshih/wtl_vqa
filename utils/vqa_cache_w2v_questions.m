function[] = vqa_cache_w2v_questions(vqa, outfile)
    num_elts = py.len(vqa.qqa);
    q_map = containers.Map('KeyType', 'int64', 'ValueType', 'any');
    num_mults = 18;
    cell_size = 1+num_mults;
    [dict, ~] = load_w2v_dict('word2vec_cache_utils/vqa_train_words.txt', 'word2vec_cache_utils/vqa_train_vecs.bin', 1);
    ids = cellfun(@double, cell(vqa.getQuesIds));
    for i = ids
        if mod(i, 100) == 0
            fprintf('%d/%d\n', i, ids(end));
        end
        strings = cell(cell_size,1);
        %% first position is for the question, next 10 are for the multiple choice
        qs = char(vqa.qqa{i}{'question'});
        [str_vec, senlen] = string2idx(qs, dict);
        strings{1} = {str_vec, senlen};

        mcs = vqa.qqa{i}{'multiple_choices'};
        for j = 1:num_mults
            [vec, senlen] = string2idx(char(mcs{j}), dict);
            strings{j+1} = {vec, senlen};

        end
        q_map(vqa.qqa{i}{'question_id'}) = strings;
    end
    save(outfile, 'q_map');
end

function[vec, len] = string2idx(str, dict)
    str_cell = sanitize_string(str);
    tf= isKey(dict, str_cell);
    vc = values(dict, str_cell(tf));
    vec = cat(1, vc{:});
    len = length(str_cell);         
end