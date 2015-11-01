% Learning method for one-class classification with KNFST according to the work:
%
% Paul Bodesheim and Alexander Freytag and Erik Rodner and Michael Kemmler and Joachim Denzler:
% "Kernel Null Space Methods for Novelty Detection".
% Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2013.
%
% Please cite that paper if you are using this code!
%
%
% function model = learn_oneClassNovelty_knfst(K)
%
% compute one-class KNFST model by separating target data from origin in feature space
%
% INPUT: 
%   K -- (n x n) kernel matrix containing pairwise similarities between n training samples
%
% OUTPUT: 
%   model -- the one-class KNFST model used in test_oneClassNovelty_knfst
%
% (LGPL) copyright by Paul Bodesheim and Alexander Freytag and Erik Rodner and Michael Kemmler and Joachim Denzler
%
function model = learn_oneClassNovelty_knfst(K)
    
    % get number of training samples
    n = size(K,1);

    % include dot products of training samples and the origin in feature space (these dot products are always zero!)
    K = [K, zeros(n,1); zeros(1,n), 0];

    % create one-class labels + a different label for the origin
    labels = [ ones(n,1) ; 0 ];
    
    % get model parameters
    model.proj = calculateKNFST(K,labels);
    model.targetValue = mean(K(labels==1,:)*model.proj);
    model.proj = model.proj(1:n,:);

end