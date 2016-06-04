function[qs] = vqa_load_test_qs()

% no test annotations
quesFile = '../data/VQA/Questions/MultipleChoice_mscoco_test2015_questions.json';
qs = py.json.load(py.open(quesFile, 'r'));