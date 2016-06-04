function[] = mscoco_cache_boxes()
globals;

mkdir(MSCOCO_BOX_CACHE_DIR);

cache_boxes(MSCOCO_VAL_DIR, MSCOCO_BOX_CACHE_DIR);
% cache the train dir
cache_boxes(MSCOCO_TRAIN_DIR, MSCOCO_BOX_CACHE_DIR);
% cache the val dir

end


function[] = cache_boxes(imdir, cachedir)
    files = dir(fullfile(imdir, '*.jpg'));
    cnt = 1;
    num_files = length(files);
    parfor cnt = 1:num_files
        fprintf('%d/%d\n', cnt, num_files);        
        resfname = fullfile(cachedir, [files(cnt).name(1:end-4), '_eb.mat']);
        if exist(resfname)
            continue;
        end
        im = imread(fullfile(imdir,files(cnt).name));
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