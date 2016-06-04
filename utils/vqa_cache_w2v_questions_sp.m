function[] = vqa_cache_w2v_questions_sp(vqa, outfile, parsefile, w2vwords, w2vword_vecs)
%% question caching function for vqa
%% Uses output of Stanford Parser
    num_elts = py.len(vqa.qqa);
    q_map_sp = containers.Map('KeyType', 'int64', 'ValueType', 'any');
    num_mults = 18;
    cell_size = 1+num_mults;
    [dict, ~] = load_w2v_dict(w2vwords,w2vword_vecs, 1);
    ids = cellfun(@double, cell(vqa.getQuesIds));
    pfile = fopen(parsefile, 'r');
    curr = 1;
    for i = ids
        if mod(curr, 100) == 0
            fprintf('%d/%d\n', curr, length(ids));
        end
        curr = curr+1;
        strings = cell(cell_size,1);
        %% first position is for the question, next 10 are for the multiple choice
        qs = char(vqa.qqa{i}{'question'});
        qs_parsed = fgets(pfile);
        p_bins = strsplit(qs_parsed, '|');
        assert(numel(p_bins) == 6);
        %        [str_vec, senlen] = string2idx(qs, dict);
        %strings{1} = {str_vec, senlen};

        prefv = string2idx(p_bins{2}, dict);
        restv = string2idx(p_bins{3}, dict);
        nsubjv = string2idx(p_bins{4}, dict);
        othernv = string2idx(p_bins{5}, dict);
        strings{1} = {prefv, restv, nsubjv, othernv};
        mcs = vqa.qqa{i}{'multiple_choices'};
        for j = 1:num_mults
            [vec, senlen] = string2idx(char(mcs{j}), dict);
            strings{j+1} = vec;

        end
        q_map_sp(vqa.qqa{i}{'question_id'}) = strings;
    end
    save(outfile, 'q_map_sp');


end

function[vec, len] = string2idx(str, dict)
    str_cell = sanitize_string(str);
    tf= isKey(dict, str_cell);
    vc = values(dict, str_cell(tf));
    vec = cat(1, vc{:});
    len = length(str_cell);         
end