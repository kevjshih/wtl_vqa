function[] = mscoco_visualize_vgg_box_categories()
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
        a = load(boxfname, 'boxes', 'scores');

        num_boxes = min(length(a.scores), num_boxes_to_keep);
        boxes = a.boxes(1:num_boxes,:);
        cropped_im = zeros(cropsize, cropsize, 3, num_boxes, 'single');
        boxes_padded = zeros(num_boxes, 4);
        parfor j = 1:num_boxes
            
            boxes_padded(j,:) = rcnn_im_crop(im, boxes(j,:), 'warp',  cropsize, padding);
            cropped_im(:,:,:,j) = get_warped_crop(im, boxes_padded(j,:), cropsize);        
        end
        cropped_im = cropped_im - avIm(:,:,:,1:num_boxes);
        res = vl_simplenn(vggnet, gpuArray(cropped_im));
        vgg_feats = gather(squeeze(res(end-2).x));
        vgg_classes =gather(squeeze(res(end).x));
        keyboard;
        % for j = 1:10%num_boxes
        %     figure(1);
        %     imshow(im/256);
        %     hold on;
        %     scores = vgg_classes(:,j);
        %     [s, idx] =max(scores);
        %     disp(vggnet.classes.description{idx});
        %     disp(['score: ' num2str(s)]);
        %     draw_bbox(boxes_padded(j,:), '-r', 'LineWidth', 3);
        %     hold off;
        %     pause
        % end
        
    end
end