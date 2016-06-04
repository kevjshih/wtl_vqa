function[vqa] = vqa_load_testdev()

% no test annotations
quesFile = '../data/VQA/Questions/MultipleChoice_mscoco_test-dev2015_questions.json';
vqa = py.vqa.VQA(pyargs('question_file', quesFile));