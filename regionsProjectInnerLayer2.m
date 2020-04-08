function res = regionsProjectInnerLayer2(layer, res_i, res_ip1, do_forward,opts)
% Takes the region features and answer features and generates a answer dependent
% representation.
%
%% weights for input projection (used to determine region weights)
% weights{1} should be size of 1 x 1 x L_feat_size x H
% weights{2} (bias) should be 1 x H
% weights{3} should be size of 1 x 1 x V_feat_size x H
% weights{4} (bias) should be 1 x H

%% weights for output projection
% weights{5} should be size of 1 x 1 x L_feat_size x G
% weights{6} (bias) should be 1 x G
% weights{7} should be size of 1 x 1 x V_feat_size x G
% weights{8} (bias) should be 1 x G
%%  weights{9} should be size of 1 x 1 x G x J
%% weights{10} (bias) should be 1 x J


% J can be anything (the next layer will get this output).
%
% Author: saurabh.me@gmail.com (Saurabh Singh) and kjshih2@illinois.edu (Kevin Shih).

% l_feats should be 1 x num_mults x L_feat_size x batchsize

l = layer;
x = res_i.x;
region_feats = l.region_feats;
% regions_feats is num_regions x 1 x num_feats x batch_size.
assert(size(region_feats, 2) == 1);
convOpts = {'CudnnWorkspaceLimit', 1024*1024*1024*2 } ;

if do_forward
    res.dzdx = [];
    res.dzdw = [];
  % Project the answer features
  resp_a_inner = vl_nnconv(x, l.weights{1}, l.weights{2}, 'pad', 0, ...
    'stride', 1, 'CuDNN',convOpts{:});
  
  % Project the region features
  resp_r_inner = vl_nnconv(region_feats, l.weights{3}, l.weights{4}, 'pad', 0, ...
                           'stride', 1, 'CuDNN',convOpts{:});

  % Combine answer and regions with an inner product
  comb_feats_inner = sum(bsxfun(@times, resp_a_inner, resp_r_inner),3);
  % Compute the weights.
  m = max(comb_feats_inner, [], 1);
  ew = exp(bsxfun(@minus, comb_feats_inner, m));
  sew = sum(ew, 1);
  nw = bsxfun(@rdivide, ew, sew);
  resp_a_outer = vl_nnconv(x, l.weights{5}, l.weights{6}, 'pad', 0, 'stride', 1, 'CuDNN',convOpts{:});
  resp_r_outer = vl_nnconv(region_feats, l.weights{7}, l.weights{8}, 'pad', 0, 'stride', 1, 'CuDNN', convOpts{:});
  % Combine the region and answer features to allow diff ans to select diff
  % regions.
  comb_feats_outer = bsxfun(@plus, resp_a_outer, resp_r_outer);
  %  comb_feats_outer_relu = comb_feats_outer .* (comb_feats_outer > 0);
  %% relu nonlinearity here
  
  % Do output projection.
  %  resp_o = vl_nnconv(comb_feats_outer_relu, l.weights{9}, l.weights{10}, 'pad', 0, ...
  %  'stride', 1);
    
  % Construct the feature
  %  weighted_feats = bsxfun(@times, resp_o, nw);
  %weighted_feats_sum = sum(weighted_feats, 1);
  %feat = weighted_feats_sum .* (weighted_feats_sum > 0);

  weighted_feats = bsxfun(@times, comb_feats_outer, nw);
  weighted_feats_sum = sum(weighted_feats, 1);
  feat = weighted_feats_sum;
  %  feat = weighted_feats_sum .* (weighted_feats_sum > 0);  
  res = res_ip1;
  res.x = feat;

  res.aux.nw = nw;
  res.aux.weighted_feats = gather(weighted_feats);
  res.aux.comb_feats_outer = gather(comb_feats_outer);
  res.aux.weighted_feats_sum = gather(weighted_feats_sum);
  res.aux.resp_a_inner = gather(resp_a_inner);
  res.aux.resp_r_inner = gather(resp_r_inner);
  res.aux.comb_feats_inner = gather(comb_feats_inner);
else
  res = res_i;
  res_x = res_ip1.x;
  nw = res_ip1.aux.nw;
  %  comb_feats_outer_relu = res_ip1.aux.comb_feats_outer_relu;
  % resp_o = res_ip1.aux.resp_o;
  comb_feats_outer = res_ip1.aux.comb_feats_outer;
  weighted_feats_sum = res_ip1.aux.weighted_feats_sum;
  resp_a_inner = res_ip1.aux.resp_a_inner;
  resp_r_inner = res_ip1.aux.resp_r_inner;
  res_dzdx = res_ip1.dzdx;
  %  res_dzdx = res_dzdx .* (res_x > 0); % undo the relu

  res_dzdx_o = bsxfun(@times, res_dzdx, nw);
    
  dzdw = cell(1, 8);
  %[dzdx_o, dzdw{9}, dzdw{10}] = vl_nnconv(comb_feats_outer_relu, l.weights{9}, ...
  %                                        l.weights{10}, res_dzdx_o, 'pad', 0, 'stride', 1);
  %dzdx_o = dzdx_o .* (comb_feats_outer_relu > 0);
  [dzdx_oa, dzdw{5}, dzdw{6}] = vl_nnconv(x, l.weights{5}, l.weights{6}, sum(res_dzdx_o,1), 'pad', 0, 'stride', 1, 'CuDNN', convOpts{:});

  [dzdx_or, dzdw{7}, dzdw{8}] = vl_nnconv(region_feats, l.weights{7}, l.weights{8}, sum(res_dzdx_o, 2), 'pad', 0, 'stride', 1,'CuDNN', convOpts{:});

  res_dzdx_nw = bsxfun(@times, res_dzdx, bsxfun(@minus, comb_feats_outer, weighted_feats_sum));

  dw = sum(res_dzdx_nw, 3).*nw;

  dzdx_a_inner_comb = sum(bsxfun(@times, dw, resp_r_inner), 1);
  dzdx_r_inner_comb = sum(bsxfun(@times, dw, resp_a_inner), 2);
  [dzdx_a_inner, dzdw{1}, dzdw{2}] = vl_nnconv(x, l.weights{1}, l.weights{2}, dzdx_a_inner_comb,...
                                               'pad', 0, 'stride', 1,'CuDNN',convOpts{:});
  [dzdx_r_inner, dzdw{3}, dzdw{4}] = vl_nnconv(region_feats, l.weights{3}, l.weights{4}, dzdx_r_inner_comb,...
                                               'pad', 0, 'stride', 1,'CuDNN',convOpts{:});
  dzdx = dzdx_a_inner + dzdx_oa;
  res.dzdx = dzdx;
  res.dzdw = dzdw;
  res.aux = [];

end
end
