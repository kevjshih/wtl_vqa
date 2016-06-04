function[vqa] = vqa_load_train()

annFile = 'data/VQA/Annotations/mscoco_train2014_annotations.json';
quesFile = 'data/VQA/Questions/MultipleChoice_mscoco_train2014_questions.json';
vqa = py.vqa.VQA(annFile, quesFile);