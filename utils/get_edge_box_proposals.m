function[bboxes, s] = get_edge_box_proposals(im, model)

% utility function for extracting edge box proposals from a single image
if nargin < 2
    model = load('external/edges/models/forest/modelBsds');
    model = model.model;
end
opts = edgeBoxes;

opts.alpha    = .65;  % step size of sliding window search
                      %opts.beta     = .75;  % nms threshold for object proposals
opts.beta = 0.2;
opts.minScore = .01;  % min score of boxes to detect
opts.maxBoxes = 3000; % max number of boxes to detect

bbs = edgeBoxes(im, model, opts);

s = bbs(:,end);
bboxes = bbs(:,1:4);
bboxes(:,3) = bboxes(:,1)+bboxes(:,3)-1;
bboxes(:,4) = bboxes(:,2)+bboxes(:,4)-1;

end