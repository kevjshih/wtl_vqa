function res = mcqMaxMarginLossLayer(layer, res_i, res_ip1, do_forward)
% Multiple choice question max margin loss layer.
%
% Loss enforces that the correct answer should score higher than the incorrect
% ones by a margin of one.
%
% Author: saurabh.me@gmail.com (Saurabh Singh).

x = res_i.x;
assert(size(x, 1) * size(x, 3) == 1, 'Size of x should be 1 x num_ans x 1 x N');

correct_answer = layer.correct_answer;
answer_size = size(correct_answer);
assert(prod(answer_size(1:3)) == 1, 'Answer size should be 1 x 1 x 1 x N');

annotator_choice_frac = layer.annotator_choice_frac;
% If annotator_choice_frac is empty then use a margin of 1 with everything.
if ~isempty(annotator_choice_frac)
  assert(all(size(x) == size(annotator_choice_frac)), ...
    'Size of annotator choice should be the same as x');
end

x_size = size(x);
num_answers = prod(x_size(1:3));
if do_forward
  res = res_ip1;
  

  answer_inds = reshape((0 : num_answers : numel(x)-1), ...
    size(correct_answer)) + correct_answer;
  correct_answer_scores = reshape(x(answer_inds), 1, 1, 1, []);
  
  % Compute the margins.
  if ~isempty(annotator_choice_frac)
    margins = computeMarginsFromAnnotatorChoices(annotator_choice_frac, ...
      answer_inds);
  else
     margins = ones(size(x), 'single');
     margins(answer_inds) = 0;
  end
  
  % Compute the loss.
  losses = max(0, bsxfun(@minus, margins + x, correct_answer_scores));
  [all_losses, max_violator] = max(losses, [], 2);

  % Compute the predictions. This is the highest scoring answer.
  [pred_score, pred] = max(x, [], 2);
  
  res.x = sum(all_losses);
  res.aux.all_losses = gather(all_losses);
  res.aux.pred = gather(pred);
  res.aux.pred_score = gather(pred_score);
  res.aux.max_violator = gather(max_violator);
  res.aux.correct_answer_inds = gather(answer_inds);
else
  res = res_i;
  max_violator = res_ip1.aux.max_violator;
  correct_answer_inds = res_ip1.aux.correct_answer_inds;
  all_losses = res_ip1.aux.all_losses;
  
  dzdx = 0 .* x;
  s = all_losses > 0;
  dzdx(correct_answer_inds(s)) = -1;

  violator_inds = reshape((0 : num_answers : numel(x)-1), 1 , 1, 1, []) ...
    + max_violator;
  dzdx(violator_inds(s)) = 1;
  
  res.dzdx = dzdx;
  res.aux = [];
end
end

function margins = computeMarginsFromAnnotatorChoices(annotator_choice_frac, ...
  answer_inds)
margins = bsxfun(@minus, ...
  reshape(annotator_choice_frac(answer_inds), 1, 1, 1, []), ...
  annotator_choice_frac);
end
