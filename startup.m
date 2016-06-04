%  add vqa tools to python path
insert(py.sys.path, int32(0), 'data/VQA/PythonHelperTools/vqaTools/')
insert(py.sys.path, int32(0), '')

% set path to matconvnet here
addpath(genpath('../matconvnet-1.0-beta19/matlab'))
vl_setupnn
addpath('utils');


