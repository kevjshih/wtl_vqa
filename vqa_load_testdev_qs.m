function[qs] = vqa_load_testdev_qs()

% no test annotations
quesFile = '../data/VQA/Questions/MultipleChoice_mscoco_test-dev2015_questions.json';
qs = py.json.load(py.open(quesFile, 'r'));