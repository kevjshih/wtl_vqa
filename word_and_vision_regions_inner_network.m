function word_and_vision_regions_inner_network()
    globals;
    opts.numFetchThreads = 3;
    opts.lite = false;
    opts.train.batchSize = 50;
    opts.train.prefetch = false;
    opts.train.sync = false;
    opts.train.learningRate = [1e-4, logspace(-2, -4, 40)*0.1];


    opts.train.plotDiagnostics = false;
    opts.train.numEpochs = numel(opts.train.learningRate);

    % specifies where model snapshots will be saved after each
    % epoch
    opts.train.expDir = fullfile('results', 'vlregion_lownms_inner_sp_glorot');
    opts.train.continue = true;
    opts.train.gpus = 1
    opts.errorFunction=[];
    imdb = [];
    imdb.vqa = vqa_load_train();
    [imdb.dict, imdb.vec_matrix] = load_w2v_dict('word2vec_cache_utils/vqa_train_words.txt',...
                                                 'word2vec_cache_utils/vqa_train_vecs.bin' , 1);
    


    qmap = load('word2vec_cache_utils/vqa_train2014_qmap_sp.mat', 'q_map_sp');
    imdb.vqa_qmap = qmap.q_map_sp;
    
    lmap = load('word2vec_cache_utils/vqa_train2014_labelmap.mat', 'label_map');
    imdb.vqa_lmap = lmap.label_map;


    %% map blist to w2v indices
    determiner_list; % load determiner blacklist
    tf = isKey(imdb.dict, determiners_blist);
    det_list_found = determiners_blist(tf);
    vs = values(imdb.dict, det_list_found);
    imdb.det_blist_vals = cat(1, vs{:}); % 

    
    imdb.ids_all = imdb.vqa.getQuesIds();
    imdb.num_regions = 100;
    imdb.vggnet = load('imagenet-vgg-s.mat') ;
    imdb.vggnet.layers = imdb.vggnet.layers(end-1:end); % truncated network
    imdb.vggnet = vl_simplenn_move2(imdb.vggnet, 'gpu');
    imdb.vggnet = vl_simplenn_tidy(imdb.vggnet);
    imdb.cachefeatpath = fullfile(MSCOCO_BOX_FEAT_CACHE_DIR, 'COCO_train2014_%012g_eb_vgg.mat');
    ids = imdb.vqa.getQuesIds(); % index of all questions
    num_questions = length(ids);
    
    ids = imdb.vqa.getQuesIds(); % index of all questions
    num_questions = length(ids);
    num_train = round(num_questions*0.9);
    train_idx = cellfun(@double, cell(ids));
    val_idx = train_idx(1+num_train:num_questions);%train_idx(1)+num_train:num_questions;
    train_idx = train_idx(1:num_train);

    
    net = vl_simplenn_tidy(word_and_vision_regions_inner_network_init(18));

    net = word_and_vision_regions_conf_train(net, imdb, @word_and_vision_regions_network_getBatch,...
                                                   opts.train, 'train', train_idx, 'val', val_idx);

        
end

