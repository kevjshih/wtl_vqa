REQUIREMENTS:
MATLAB 2015a or newer (for native python support)
MatConvNet beta19 http://www.vlfeat.org/matconvnet/

SETUP INSTRUCTIONS:
1. Create a data directory
2. Setup directory paths as specified in globals.m. Relevant tarballs specified in comments
3. Clone the VQA API into the data directory and setup annotations 
4. Install matconvnet beta19 (17 or higher should work) and specify path in startup.m (with CuDNN enabled)
5. Download the vgg-s model from: http://www.vlfeat.org/matconvnet/models/imagenet-vgg-s.mat
6. Create results directory to store model snapshots from training
7. Download and extract text feature caches to top level directory (wget http://xor.cs.illinois.edu/~kevin/wtl_cache_feats/word2vec_cache_utils.tar.gz)

MAIN FUNCTIONS:
word_and_vision_regions_inner_network.m : running this should initialize training. Results stored in opts.train.expDir
word_and_vision_regions_inner_network_init.m: constructs the network
mcqMaxMarginLossLayer.m: Loss layer implementation
regionsProjectInnerLayer2.m: region selection layer implementation
determiner_list.m: list of removed stopwords removed from questions
globals.m: contains global paths to where cached features are stored.

VISUALIZATION EXAMPLE:
run visualize_on_held_out.m to visualize results on the held-out set. The held out set comprises 10% of the training data from the train set.
Our test model can be downloaded from:  http://xor.cs.illinois.edu/~kevin/wtl_cache_feats/wtl_trainval_model.mat

DIRECTORIES:
word2vec_cache_utils: directory that holds caches of pre-processed question and answers
utils: misc utility functions 


This code is provided for academic use only.

If you have any questions about the code, feel free to contact Kevin Shih at kjshih2@illinois.edu.



