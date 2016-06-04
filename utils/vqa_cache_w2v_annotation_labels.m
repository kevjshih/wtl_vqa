function[] = vqa_cache_w2v_annotation_labels(vqa, outfile)
    ids = vqa.getQuesIds();
    idx = cellfun(@double, cell(ids));
    ann = vqa.loadQA(ids);
    num_responses = 10;
    num_mults = 18;
    label_map = containers.Map('KeyType', 'int64', 'ValueType', 'any');
    for i = 1:length(idx)
        annotator_choice_frac = zeros(num_mults,1);
        if mod(i, 100) == 0
            fprintf('%d/%d\n', idx(i), idx(end));
        end
        qs_ann = vqa.qqa{ann{i}{'question_id'}};
        ans_str = cell(num_responses,1);

        for j = 1:num_responses
            ans_str{j} = char(ann{i}{'answers'}{j}{'answer'});
        end
        [unique_strs, ~, str_map] = unique(ans_str);
        as = unique_strs{mode(str_map)};
        mult_as = qs_ann{'multiple_choices'};
        %% parse multiple choices
        correct_idx = -1;        
        for j = 1:num_mults
            as_mult = char(mult_as{j});
            ind  = find(ismember(unique_strs,as_mult));
            if ~isempty(ind)
                num_occurences = sum(str_map == ind);
                annotator_choice_frac(j) = num_occurences/num_responses;
            end
            if strcmp(as, as_mult) == 1
                correct_idx = j;
            end
        end
        cnt = 1;
        while(correct_idx < 0)
            as = unique_strs{cnt};
            for j = 1:num_mults
                as_mult = char(mult_as{j});
                if strcmp(as, as_mult) == 1
                    correct_idx = j;
                end
            end
            cnt= cnt +1;            
        end                    
        label_map(ann{i}{'question_id'}) = {correct_idx, annotator_choice_frac};
    end
    save(outfile, 'label_map');
end