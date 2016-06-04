function[] = vqa_visualize_results(res_map, is_train)
    if is_train
        vqa = vqa_load_train
        impath = '../data/MSCOCO/train2014/COCO_train2014_%012g.jpg';
    else
        vqa = vqa_load_val
        impath = '../data/MSCOCO/val2014/COCO_val2014_%012g.jpg';
    end


    q_ids = keys(res_map);
    for i = 1:length(q_ids)
        qs = vqa.qqa{q_ids{i}};
        fprintf('Q: %s\n',char(qs{'question'}));
        mcs = cell(qs{'multiple_choices'});
        for j = 1:length(mcs)
            fprintf('\t* %s\n', char(mcs{j}));
        end

        imshow(sprintf(impath, qs{'image_id'}));
        fprintf('\nResponse: %s\n', res_map(q_ids{i}));
        pause();
    end
    
    
end