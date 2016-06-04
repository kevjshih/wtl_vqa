function[] = word_and_vision_regions_network_visualize_regions(snapshot_file, on_train)
    if nargin < 2
        on_train = 0;
    end

    globals;
    opts.numFetchThreads = 3;
    opts.lite = false;
    opts.train.batchSize = 40;
    opts.train.prefetch = false;
    opts.train.sync = false;
    opts.train.continue = false;
    opts.train.numSubBatches = 1
    opts.train.conserveMemory = false ;
    opts.train.backPropDepth = +inf ;
    opts.train.plotDiagnostics = false ;        
    opts.train.gpus = 1;
    % setup GPUs
    numGpus = numel(opts.train.gpus) ;
    if numGpus > 1
        if isempty(gcp('nocreate')),
            parpool('local',numGpus) ;
            spmd, gpuDevice(opts.gpus(labindex)), end
        end
    elseif numGpus == 1
        gpuDevice(opts.train.gpus)
    end
    load(snapshot_file, 'net')
    net = vl_simplenn_move2(net, 'gpu');

    imdb = [];

    if on_train == 0 % val set
                     % load val set
        imdb.vqa = vqa_load_val();
        qmap = load('word2vec_cache_utils/vqa_val2014_qmap_sp.mat', 'q_map_sp');
        imdb.vqa_qmap = qmap.q_map_sp;
        
        lmap = load('word2vec_cache_utils/vqa_val2014_labelmap.mat', 'label_map');
        imdb.impath = 'data/MSCOCO/val2014/COCO_val2014_%012g.jpg';
        [imdb.dict, imdb.vec_matrix] = load_w2v_dict('word2vec_cache_utils/vqa_val_words.txt', 'word2vec_cache_utils/vqa_val_vecs.bin' , 1);
    else
        % load train set
        imdb.vqa = vqa_load_train();
        qmap = load('word2vec_cache_utils/vqa_train2014_qmap_sp.mat', 'q_map_sp');
        imdb.vqa_qmap = qmap.q_map_sp;
        
        
        lmap = load('word2vec_cache_utils/vqa_train2014_labelmap.mat', 'label_map');
        imdb.impath = 'data/MSCOCO/train2014/COCO_train2014_%012g.jpg';
        [imdb.dict, imdb.vec_matrix] = load_w2v_dict('word2vec_cache_utils/vqa_train_words.txt', 'word2vec_cache_utils/vqa_train_vecs.bin' , 1);
    end
    imdb.vggnet = load('imagenet-vgg-s.mat') ;
    imdb.vggnet.layers = imdb.vggnet.layers(end-1:end); % truncated network
    imdb.vggnet = vl_simplenn_tidy(vl_simplenn_move2(imdb.vggnet, 'gpu'));
    
    imdb.vqa_lmap = lmap.label_map;    
    %imdb.dict = load_w2v_dict('word2vec_cache_utils/vqa_train_words.txt', 'word2vec_cache_utils/vqa_train_vecs.bin',1);
    
    imdb.cardinal_ids = imdb.vqa.getQuesIds(pyargs('quesTypes','how many'));
    imdb.ids_all = imdb.vqa.getQuesIds();
    imdb.num_regions = 100;
    ids = imdb.vqa.getQuesIds;
    num_questions = length(ids);

    
    
    if on_train >0
        num_train = round(num_questions*0.9);
        train_idx = cellfun(@double, cell(ids));

        if on_train == 1
            val_idx = train_idx(1+num_train:num_questions);
        else
            %            val_idx = train_idx(1):train_idx(1)+round(num_train/20);
        end
        imdb.cachefeatpath = fullfile(MSCOCO_BOX_FEAT_CACHE_DIR, 'COCO_train2014_%012g_eb_vgg.mat');
    else
        val_idx = cellfun(@double, cell(ids));
        imdb.cachefeatpath = fullfile(MSCOCO_BOX_FEAT_CACHE_DIR, 'COCO_val2014_%012g_eb_vgg.mat');
    end
    %    val_idx = val_idx(1:1000);
    %if exist(opts.memoryMapFile), delete(opts.memoryMapFile) ; end

    %% map blist to w2v indices
    determiner_list; % load determiner blacklist
    tf = isKey(imdb.dict, determiners_blist);
    det_list_found = determiners_blist(tf);
    vs = values(imdb.dict, det_list_found);
    imdb.det_blist_vals = cat(1, vs{:}); % 

    [res_map] = process_epoch(opts.train, @word_and_vision_regions_network_getBatch, 1, val_idx, 0, imdb, net) ;
end


% -------------------------------------------------------------------------
function  [res_map] = process_epoch(opts, getBatch, epoch, subset, learningRate, imdb, net)
% -------------------------------------------------------------------------
    res_map = containers.Map('KeyType', 'int64', 'ValueType', 'any');
    % validation mode if learning rate is zero
    training = learningRate > 0 ;
    if training, mode = 'training' ; else, mode = 'validation' ; end
    if nargout > 2, mpiprofile on ; end

    numGpus = 1;
    if numGpus >= 1
        one = gpuArray(single(1)) ;
    else
        one = single(1) ;
    end
    res = [] ;
    mmap = [] ;
    stats = [] ;
    num_correct = 0;
    num_total = 0;

    % revert later!
    for t = 1:opts.batchSize:numel(subset)
        fprintf('%s: epoch %02d: batch %3d/%3d: ', mode, epoch, ...
                fix(t/opts.batchSize)+1, ceil(numel(subset)/opts.batchSize)) ;
        batchSize = min(opts.batchSize, numel(subset) - t + 1) ;
        batchTime = tic ;
        numDone = 0 ;
        error = [] ;
        for s=1:opts.numSubBatches
            % get this image batch and prefetch the next
            batchStart = t + (labindex-1) + (s-1) * numlabs ;
            batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
            batch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
            [im, labels, annotator_choice_frac, region_feats, qs_anns] = getBatch(imdb, batch) ;

            if opts.prefetch
                if s==opts.numSubBatches
                    batchStart = t + (labindex-1) + opts.batchSize ;
                    batchEnd = min(t+2*opts.batchSize-1, numel(subset)) ;
                else
                    batchStart = batchStart + numlabs ;
                end
                nextBatch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
                getBatch(imdb, nextBatch) ;
            end

            if numGpus >= 1
                im = gpuArray(im) ;
            end

            % evaluate CNN
            net.layers{end}.class = labels ;
            if training, dzdy = one; else, dzdy = [] ; end
            net.layers{end}.correct_answer = labels;
            net.layers{end}.annotator_choice_frac = annotator_choice_frac;
            net.layers{end-7}.region_feats = gpuArray(region_feats);
            res = vl_simplenn(net, im, dzdy, res, ...
                              'accumulate', s ~= 1, ...
                              'mode', 'test',...
                              'cudnn', true,...
                              'conserveMemory', opts.conserveMemory, ...
                              'backPropDepth', opts.backPropDepth, ...
                              'sync', opts.sync) ;

            %% fill in answers into empty curr_res_map
            preds_cpu= gather(res(end).aux.pred);
            pred_scores = squeeze(gather(res(end-1).x));
            for q = 1:length(qs_anns)
                disp(q);

                confs = single(pred_scores(:,q));
                %                res_map(qs_anns{q}{'question_id'}) = char(qs_anns{q}{'multiple_choices'}{preds_cpu(q)});
                res_map(qs_anns{q}{'question_id'}) = confs;
                im_name = fullfile(sprintf(imdb.impath, qs_anns{q}{'image_id'}));


                
                region_feat_cache_file = sprintf(imdb.cachefeatpath, qs_anns{q}{'image_id'});
                load(region_feat_cache_file, 'vgg_feats', 'boxes_padded');
                boxes = load(region_feat_cache_file, 'boxes_padded');
                boxes.boxes_padded(boxes.boxes_padded< 1) = 1;
                nw = gather(res(11).aux.nw(:,:,:,q));

                im_orig = im2double(imread(im_name));
                [h, w, d] = size(im_orig);
                if d < 3
                    im_orig = repmat(im_orig, [ 1 1 3]);
                end
                j = preds_cpu(q)

                
                im = im_orig;
                num_boxes = size(boxes.boxes_padded,1);
                mask = zeros(h,w);   
                wts = nw(:,j);

                wts = wts(1:num_boxes);
                
                [wts_srtd, srtd_idx] = sort(wts, 'descend');                                                
                for z = 1:num_boxes
                    mask(boxes.boxes_padded(z,2):min(boxes.boxes_padded(z,4), h), min(w,boxes.boxes_padded(z,1):boxes.boxes_padded(z,3))) =  mask(boxes.boxes_padded(z,2):min(h,boxes.boxes_padded(z,4)), min(w,boxes.boxes_padded(z,1):boxes.boxes_padded(z,3))) +wts(z);
                end
                mask = imgaussfilt(mask, 15);
                mask = mask./max(mask(:));
                im = im.*repmat(mask, [1 1 3]);
                toks = strsplit(im_name, '/');
                
                imshow(im)
                disp(char(qs_anns{q}{'question'}))
                [scr, scr_idx] = sort(confs, 'descend');
                for jj = 1:18
                    fprintf('%.3g\t', scr(jj));
                    fprintf(char(qs_anns{q}{'multiple_choices'}{scr_idx(jj)}));
                    fprintf('\n')

                end
                                       
                pause()

            end

            error= sum([error,res(end).x]);


            numDone = numDone + numel(batch) ;
        end

        % print learning statistics
        batchTime = toc(batchTime) ;
        stats = sum([stats,[batchTime ; error]],2); % works even when stats=[]
        speed = batchSize/batchTime ;

        fprintf(' %.2f s (%.1f data/s)', batchTime, speed) ;
        n = (t + batchSize - 1) / max(1,numlabs) ;
        fprintf(' obj:%.3g', stats(2)/n) ;
        %  for i=1:numel(opts.errorLabels)
        %  fprintf(' %s:%.3g', opts.errorLabels{i}, stats(i+2)/n) ;
        %end
        fprintf(' [%d/%d]', numDone, batchSize);
        fprintf('\n') ;

        % debug info
        if opts.plotDiagnostics && numGpus <= 1
            figure(2) ; vl_simplenn_diagnose(net,res) ; drawnow ;
        end
    end

    if nargout > 2
        prof = mpiprofile('info');
        mpiprofile off ;
    end
    %    stats.num_correct = num_correct;
    %stats.num_total = num_total;
end
