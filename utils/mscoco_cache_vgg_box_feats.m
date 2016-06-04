function[] = mscoco_cache_vgg_box_feats()
    globals
    mkdir(MSCOCO_BOX_FEAT_CACHE_DIR);
    vggnet = load('imagenet-vgg-s.mat');
    vggnet = vl_simplenn_move(vggnet, 'gpu');
    cache_box_feats(MSCOCO_VAL_DIR, MSCOCO_BOX_CACHE_DIR, MSCOCO_BOX_FEAT_CACHE_DIR, vggnet);
    cache_box_feats(MSCOCO_TRAIN_DIR, MSCOCO_BOX_CACHE_DIR, MSCOCO_BOX_FEAT_CACHE_DIR, vggnet);


end


function[] = cache_box_feats(imdir, boxdir, featcachedir, vggnet)
    files = dir(fullfile(imdir, '*.jpg'));    
    num_files = length(files);
    num_boxes_to_keep = 100;
    cropsize = 224;
    padding = 18;
    avIm = repmat(vggnet.normalization.averageImage, [1,1, 1, num_boxes_to_keep]);
    for i = 1:num_files
        if mod(i, 50) == 0
            fprintf('%d/%d\n', i ,num_files);
        end
        im = single(imread(fullfile(imdir, files(i).name)));
        boxfname = fullfile(boxdir, [files(i).name(1:end-4), '_eb.mat']);
        resfname = fullfile(featcachedir, [files(i).name(1:end-4), '_eb_vgg.mat']);
        if exist(resfname) 
            continue;
        end

        a = load(boxfname, 'boxes', 'scores');

        num_boxes = min(length(a.scores), num_boxes_to_keep);
        boxes = a.boxes(1:num_boxes-1,:);
        [r, c, d] = size(im);
        boxes(end+1,:) = [1 1 c r];
        cropped_im = zeros(cropsize, cropsize, 3, num_boxes, 'single');
        boxes_padded = zeros(num_boxes, 4);
        parfor j = 1:num_boxes            
            boxes_padded(j,:) = rcnn_im_crop(im, boxes(j,:), 'warp',  cropsize, padding);
            cropped_im(:,:,:,j) = get_warped_crop(im, boxes_padded(j,:), cropsize);        
        end
        cropped_im = cropped_im - avIm(:,:,:,1:num_boxes);
        res = vl_simplenn(vggnet, gpuArray(cropped_im));
        vgg_feats = gather(squeeze(res(end-2).x));
        %vgg_class_vals = gather(squeeze(res(end-1).x));
        save(resfname, 'vgg_feats', 'boxes_padded');
    end
end