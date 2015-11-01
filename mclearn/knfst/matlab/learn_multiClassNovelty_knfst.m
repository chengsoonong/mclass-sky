% Learning method for multi-class novelty detection with KNFST according to the work:
%
% Paul Bodesheim and Alexander Freytag and Erik Rodner and Michael Kemmler and Joachim Denzler:
% "Kernel Null Space Methods for Novelty Detection".
% Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2013.
%
% Please cite that paper if you are using this code!
%
%
% function model = learn_multiClassNovelty_knfst(K, labels)
%
% calculate multi-class KNFST model for multi-class novelty detection
%
% INPUT: 
%   K -- (n x n) kernel matrix containing similarities of n training samples
%   labels -- (n x 1) column vector containing (multi-class) labels of n training samples
%
% OUTPUT:
%   model -- the multi-class KNFST model used in test_multiClassNovelty_knfst
%
% (LGPL) copyright by Paul Bodesheim and Alexander Freytag and Erik Rodner and Michael Kemmler and Joachim Denzler
%
function model = learn_multiClassNovelty_knfst(K, labels)

    % obtain unique class labels
    classes = unique(labels);

    % calculate projection of KNFST
    model.proj = calculateKNFST(K, labels);

    % calculate target points ( = projections of training data into the null space)
    model.target_points = zeros( length(classes),size(model.proj,2) );
    for c=1:length(classes)

      id = labels == classes(c);
      model.target_points(c,:) = mean(K(id,:)*model.proj); 
       
    end

end