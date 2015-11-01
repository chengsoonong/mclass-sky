% Test method for one-class classification with KNFST according to the work:
%
% Paul Bodesheim and Alexander Freytag and Erik Rodner and Michael Kemmler and Joachim Denzler:
% "Kernel Null Space Methods for Novelty Detection".
% Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2013.
%
% Please cite that paper if you are using this code!
%
%
% function scores = test_oneClassNovelty_knfst_artificialClass(model, Ks_artificial)
%
% compute novelty scores using the one-class KNFST model obtained from learn_oneClassNovelty_knfst_artificialClass
%
% INPUT: 
%    model -- model obtained from learn_oneClassNovelty_knfst_artificialClass
%    Ks_artificial    -- (2n x m) kernel matrix containing similarities of m test samples to n training samples X in the first n rows and
%                        the similarities to the negative replicates -X in the last n rows 
%
% OUTPUT: 
%   scores -- novelty scores for the test samples (distances in the null space)
%
%
% (LGPL) copyright by Paul Bodesheim and Alexander Freytag and Erik Rodner and Michael Kemmler and Joachim Denzler
%
function scores = test_oneClassNovelty_knfst_artificialClass(model, Ks_artificial)

  % projected test samples:
  projectionVectors = Ks_artificial'*model.proj;
  
  % differences to the target value:
  diff = projectionVectors-ones(size(Ks_artificial,2),1)*model.targetValue;

  % distances to the target value:
  scores = sqrt(sum(diff.*diff,2));

end 
