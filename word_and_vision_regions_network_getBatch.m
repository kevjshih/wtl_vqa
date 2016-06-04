function [input,labels, annotator_choice_frac, region_feats, qs_ann_out] = word_and_vision_regions_network_getBatch(imdb, batch)
% -------------------------------------------------------------------------
%
    num_responses = 10;
    num_mults = 18;
    ann = cell(imdb.vqa.loadQA(py.list(batch)));
    input = zeros(1,num_mults,1500,length(batch), 'single');
    labels = zeros(1,1,1,length(batch), 'single')-1;
    annotator_choice_frac = zeros(1, num_mults, 1, length(batch), 'single');
    qs_ann_out = cell(length(batch),1);

    %% load vision features
    %    region_feats = zeros(imdb.num_regions, 1, 4096, length(batch), 'single');
    region_feats = zeros(4096, imdb.num_regions, length(batch), 'single');
    class_feats = zeros(1000, imdb.num_regions, length(batch), 'single', 'gpuArray');
    for i = 1:length(batch)
        region_feat_cache_file = sprintf(imdb.cachefeatpath, ann{i}{'image_id'});
        load(region_feat_cache_file, 'vgg_feats', 'boxes_padded');
        curr_num_regions = size(vgg_feats,2);
        res2= vl_simplenn(imdb.vggnet, gpuArray(reshape(vgg_feats, [1 1 4096 curr_num_regions])));
        class_feats(:,1:curr_num_regions,i) = squeeze(res2(end-1).x); % 1000 x 100        
        region_feats(1:4096,1:curr_num_regions,i) = vgg_feats;


        %% parse question prompt
        qs_ann = imdb.vqa.qqa{ann{i}{'question_id'}};
        sents = imdb.vqa_qmap(ann{i}{'question_id'});

        %% do the parsed bucketting
        qs_idxs = sents{1};
        qs_pref_vec = mean(imdb.vec_matrix(:, qs_idxs{1}),2);
        
        qs_idxs{2} = setdiff(qs_idxs{2}, imdb.det_blist_vals);
        if ~isempty(qs_idxs{2})
            qs_other_vec = mean(imdb.vec_matrix(:, qs_idxs{2}),2);
        else
            qs_other_vec = zeros(300,1);
        end
        if ~isempty(qs_idxs{3})
            qs_subjn_vec = mean(imdb.vec_matrix(:, qs_idxs{3}),2);
        else
            qs_subjn_vec = zeros(300,1);
        end
        if ~isempty(qs_idxs{4})
            qs_othern_vec = mean(imdb.vec_matrix(:, qs_idxs{4}),2);
        else
            qs_othern_vec = zeros(300,1);
        end

        %        qs_vec = mean(imdb.vec_matrix(:, sents{1}),2);

        qs_ann_out{i} = qs_ann;
        %% parse multiple choices
        curr_labels = imdb.vqa_lmap(ann{i}{'question_id'});
        annotator_choice_frac(1,:,1,i) = reshape(curr_labels{2}, [1, num_mults, 1,1]);
        labels(i) = curr_labels{1};
        % correct_idx = -1;        
        for j = 1:num_mults            
            if isempty(sents{j+1})
                as_mult_vec = zeros(300,1, 'single');
            else
                as_mult_vec = mean(imdb.vec_matrix(:,sents{j+1}),2);
            end
            input(1,j, :,i) = reshape([qs_pref_vec(:); qs_other_vec(:); qs_subjn_vec(:); qs_othern_vec(:); as_mult_vec(:)], [1, 1, 1500]);
            %            input(1,j, :,i) = reshape([qs_vec(:); as_mult_vec(:)], [1, 1, 600]);
        end
    end
    % needs to be num_regions x 1 x num_feats x batch_size
    % temp removing class feats
    region_feats = cat(1, gpuArray(region_feats), class_feats);
    %region_feats = cat(1, gpuArray(region_feats));
    region_feats = permute(region_feats, [2 4 1 3]);

end
