% Learning method for one-class classification with KNFST according to the work:
%
% Paul Bodesheim and Alexander Freytag and Erik Rodner and Michael Kemmler and Joachim Denzler:
% "Kernel Null Space Methods for Novelty Detection".
% Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2013.
%
% Please cite that paper if you are using this code!
%
%
% function model = learn_oneClassNovelty_knfst_artificialClass(K_artificial)
%
% calculates one-class KNFST model by separating target data from minus data
%
% INPUT: 
%   K_artificial -- (2n x 2n) kernel matrix of pairwise similarities between n training samples X and their negative replicates -X !!
%                   this function assumes that the first n rows/columns of K_artificial correspond to the true samples and 
%                   the last n rows/columns of K_artificial to the negative replicates
%
% OUTPUT: 
%   model -- the one-class KNFST model used in test_oneClassNovelty_knfst_artificialClass
%
% (LGPL) copyright by Paul Bodesheim and Alexander Freytag and Erik Rodner and Michael Kemmler and Joachim Denzler
%
function model = learn_oneClassNovelty_knfst_artificialClass(K_artificial)

    % get number of training samples
    n = size(K_artificial,1)/2;
    
    % create binary labels
    labels = [ ones(n,1); -ones(n,1) ];
    
    % get model parameters
    model.proj = calculateKNFST(K_artificial,labels);
    model.targetValue = mean(K_artificial(labels==1,:)*model.proj);

end