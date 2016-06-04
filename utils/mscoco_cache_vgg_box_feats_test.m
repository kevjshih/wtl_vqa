function[] = mscoco_cache_vgg_box_feats_test(on_dev)
    globals
    mkdir(MSCOCO_TEST_BOX_FEAT_CACHE_DIR);
    vggnet = load('imagenet-vgg-s.mat');
    vggnet = vl_simplenn_move(vggnet, 'gpu');
    if on_dev
        fprintf('on test dev\n')
        im_id_str = 'test_dev_im_ids.mat'
    else
        fprintf('on test std\n')
        im_id_str = 'test_im_ids.mat'
    end
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
    ids = unique(ids);
    cache_box_feats(MSCOCO_TEST_DIR, MSCOCO_TEST_BOX_CACHE_DIR, MSCOCO_TEST_BOX_FEAT_CACHE_DIR, vggnet, ids);


end


function[] = cache_box_feats(imdir, boxdir, featcachedir, vggnet, ids)

    num_boxes_to_keep = 100;
    cropsize = 224;
    padding = 18;
    avIm = repmat(vggnet.normalization.averageImage, [1,1, 1, num_boxes_to_keep]);
    for i = 1:length(ids)
        if mod(i, 50) == 0
            fprintf('%d/%d\n', i ,length(ids));
        end
        im_id = ids(i);
        im = single(imread(fullfile(imdir, sprintf('COCO_test2015_%012d.jpg',im_id))));
        [r, c, d] = size(im);
        if d < 3
            im = repmat(im, [1 1 3]);
        end


        boxfname = fullfile(boxdir, [sprintf('COCO_test2015_%012d',im_id) , '_eb.mat']);

        resfname = fullfile(featcachedir, [sprintf('COCO_test2015_%012d',im_id) , '_eb_vgg.mat']);

        if exist(resfname) 
            continue;
        end

        a = load(boxfname, 'boxes', 'scores');

        num_boxes = min(length(a.scores), num_boxes_to_keep);
        if num_boxes == num_boxes_to_keep
            boxes = a.boxes(1:num_boxes-1,:);
        else
            boxes = a.boxes(1:num_boxes,:);
        end
        [r, c, d] = size(im);
        boxes(end+1,:) = [1 1 c r];
        num_boxes = size(boxes,1);

        cropped_im = zeros(cropsize, cropsize, 3, num_boxes, 'single');
        boxes_padded = zeros(num_boxes, 4);

        parfor j = 1:num_boxes            
            boxes_padded(j,:) = rcnn_im_crop(im, boxes(j,:), 'warp',  cropsize, padding);
            cropped_im(:,:,:,j) = get_warped_crop(im, boxes_padded(j,:), cropsize);        
        end
        cropped_im = cropped_im - avIm(:,:,:,1:num_boxes);
        res = vl_simplenn(vggnet, gpuArray(cropped_im));
        vgg_feats = gather(squeeze(res(end-2).x));
        %     vgg_class_vals = gather(squeeze(res(end-1).x));
        save(resfname, 'vgg_feats', 'boxes_padded');
    end
end