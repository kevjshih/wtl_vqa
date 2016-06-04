function[] = vqa_evaluate_accuracy(imdb, answers, ids)
answers = squeeze(answers);
    qs_anns = cell(imdb.vqa.loadQA(py.list(ids)));
    sens = zeros(1,1,300,length(ids), 'single');
    labels = zeros(300,length(ids), 'single');

    for i = 1:length(ids)
        all_as_vecs = zeros(300,10);
        qs = char(qs_anns{i}{'question'});
        % pick mode answer
        ans_str = cell(10,1);
        qs_anns{i}{'multiple_choices'}
        for j = 1:10
            ans_str{j} = char(qs_anns{i}{'answers'}{randi(j)}{'answer'});
            as = ans_str{j};
            as_vecs = string_to_vecs(as, imdb.dict);
            as_vecs_drop = min(as_vecs) == 0 & max(as_vecs) == 0;
            as_vecs(:, as_vecs_drop) = [];
            
            % take the mean across horizontal dimension of remaining vectors            
            if ~isempty(as_vecs)
            as_vec = mean(as_vecs,2);
            else
                as_vec = zeros(300,1);
            end
            
            all_as_vecs(:,j) = as_vec;
        end
        mcs = qs_anns{i}{'multiple_choices'}
        all_as_vecs_choices = zeros(300,length(mcs));

        for j = 1:length(mcs)
            ans_str{j} = char(qs_anns{i}{'multiple_choices'}{j});
            as = ans_str{j};
            as_vecs = string_to_vecs(as, imdb.dict);
            as_vecs_drop = min(as_vecs) == 0 & max(as_vecs) == 0;
            as_vecs(:, as_vecs_drop) = [];
            
            % take the mean across horizontal dimension of remaining vectors            
            if ~isempty(as_vecs)
            as_vec = mean(as_vecs,2);
            else
                as_vec = zeros(300,1);
            end
            
            all_as_vecs_choices(:,j) = as_vec;
        end

        all_dists = sum((repmat(answers(:,i), [1, 10]) - all_as_vecs).^2,1);
        all_dists_choices = sum((repmat(answers(:,i), [1, length(mcs)]) - all_as_vecs_choices).^2,1);
        [unique_strs, ~, str_map] = unique(ans_str);
        as = unique_strs{mode(str_map)};
        % convert the string to column matrix of w2v vectors
        qs_vecs = string_to_vecs(qs, imdb.dict);
        as_vecs = string_to_vecs(as, imdb.dict);

        % check and remove columns of all 0s
        qs_vecs_drop = min(qs_vecs) == 0 & max(qs_vecs) == 0;
        qs_vecs(:, qs_vecs_drop) = [];
        as_vecs_drop = min(as_vecs) == 0 & max(as_vecs) == 0;
        as_vecs(:, as_vecs_drop) = [];
        
        % take the mean across horizontal dimension of remaining vectors
        if ~isempty(qs_vecs)
            qs_vec = mean(qs_vecs,2);
        else
            qs_vec = zeros(300,1);
        end
        if ~isempty(as_vecs)
            as_vec = mean(as_vecs,2);
        else
            as_vec = zeros(300,1);
        end

        sens(1,1,:,i) = qs_vec;
        labels(:,i) = as_vec;
        keyboard;
    end

end