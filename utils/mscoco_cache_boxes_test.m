function[] = mscoco_cache_boxes_test(on_dev)
    globals;
    if on_dev
        fprintf('on test dev\n')
        qs = vqa_load_testdev_qs();
        im_id_str = 'test_dev_im_ids.mat'
    else
        fprintf('on test std\n')
        qs = vqa_load_test_qs();
        im_id_str = 'test_im_ids.mat'
    end
    num_qs = py.len(qs{'questions'});
    ids = zeros(num_qs,1, 'int64');
    disp('caching imids')


    if ~exist(im_id_str)
        for i = 1:num_qs
            if mod(i, 100) == 0
                fprintf('%d/%d\n', i, num_qs);
            end
            
            ids(i) = qs{'questions'}{i}{'image_id'};
        end
        save(im_id_str, 'ids');
    else
        load(im_id_str);
    end
    disp('caching boxes')
    cache_boxes(MSCOCO_TEST_DIR, MSCOCO_TEST_BOX_CACHE_DIR, ids);
end

function[] = cache_boxes(im_dir, cache_dir, ids)
    num_qs = length(ids);
    parfor i = 1:length(ids)
        if mod(i, 100) == 0
            fprintf('%d/%d\n', i, num_qs);
        end
        im_id = ids(i);
        %        test2015/COCO_test2015_000000199115.jpg 
        resfname = fullfile(cache_dir, [sprintf('COCO_test2015_%012d',im_id) , '_eb.mat']);
        if exist(resfname)
            continue;
        end
        im = imread(fullfile(im_dir, sprintf('COCO_test2015_%012d.jpg',im_id)));
        [r, c, d] = size(im);
        if d < 3
            im = repmat(im, [1 1 3]);
        end
        [boxes, scores] = get_edge_box_proposals(im);
        num_boxes = length(scores);
        num_kept = min(num_boxes, 2500);
        boxes = boxes(1:num_kept,:);
        scores = scores(1:num_kept);
        save_wrapper(resfname, boxes, scores);
    end
end
function[] = save_wrapper(resfname, boxes, scores)
    save(resfname, 'boxes', 'scores');

end