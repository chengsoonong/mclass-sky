% Test method for multi-class novelty detection with KNFST according to the work:
%
% Paul Bodesheim and Alexander Freytag and Erik Rodner and Michael Kemmler and Joachim Denzler:
% "Kernel Null Space Methods for Novelty Detection".
% Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2013.
%
% Please cite that paper if you are using this code!
%
%
% function scores = test_multiClassNovelty_knfst(model, Ks)
%
% compute novelty scores using the multi-class KNFST model obtained from learn_multiClassNovelty_knfst
%
% INPUT:
%    model -- model obtained from learn_multiClassNovelty_knfst
%    Ks -- (n x m) kernel matrix containing similarities between n training samples and m test samples
%
% OUTPUT:
%    scores -- novelty scores for the m test samples (distances in the null space)
%
% (LGPL) copyright by Paul Bodesheim and Alexander Freytag and Erik Rodner and Michael Kemmler and Joachim Denzler
%
function scores = test_multiClassNovelty_knfst(model, Ks)

  % projected test samples:
  projectionVectors = transpose(Ks)*model.proj;

  % squared euclidean distances to target points:
  squared_distances = squared_euclidean_distances(projectionVectors,model.target_points);

  % novelty scores as minimum distance to one of the target points 
  scores = sqrt( min(squared_distances, [], 2) );

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function distmat = squared_euclidean_distances(x,y)
% function distmat=squared_euclidean_distances(x,y)
%
% computes squared euclidean distances between data points in the rows of x and y
%
% INPUT:
%       x -- (n x d) matrix containing n samples of dimension d in its rows
%       y -- (m x d) matrix containing m samples of dimension d in its rows
%
% OUTPUT:
%       distmat -- (n x m) matrix of pairwise squared euclidean distances 
%

    distmat = zeros( size(x,1), size(y,1) );
    for i=1:size(x,1)
        for j=1:size(y,1)
            buff=(x(i,:)-y(j,:));   
            distmat(i,j)=buff*buff';
        end
    end

end