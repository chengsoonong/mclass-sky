% Test method for one-class classification with KNFST according to the work:
%
% Paul Bodesheim and Alexander Freytag and Erik Rodner and Michael Kemmler and Joachim Denzler:
% "Kernel Null Space Methods for Novelty Detection".
% Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2013.
%
% Please cite that paper if you are using this code!
%
%
% function scores = test_oneClassNovelty_knfst(model, Ks)
%
% compute novelty scores using the one-class KNFST model obtained from learn_oneClassNovelty_knfst
%
% INPUT: 
%    model -- model obtained from learn_oneClassNovelty_knfst
%    Ks    -- (n x m) kernel matrix containing similarities between n training samples and m test samples
%
% OUTPUT: 
%   scores -- novelty scores for the test samples (distances in the null space)
%
%
% (LGPL) copyright by Paul Bodesheim and Alexander Freytag and Erik Rodner and Michael Kemmler and Joachim Denzler
%
function scores = test_oneClassNovelty_knfst(model, Ks)

  % projected test samples:
  projectionVectors = Ks'*model.proj;
  
  % differences to the target value:
  diff = projectionVectors-ones(size(Ks,2),1)*model.targetValue;

  % distances to the target value:
  scores = sqrt(sum(diff.*diff,2));

end 
