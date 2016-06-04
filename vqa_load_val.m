function[vqa] = vqa_load_val()

annFile = '../data/VQA/Annotations/mscoco_val2014_annotations.json';
quesFile = '../data/VQA/Questions/MultipleChoice_mscoco_val2014_questions.json';
vqa = py.vqa.VQA(annFile, quesFile);