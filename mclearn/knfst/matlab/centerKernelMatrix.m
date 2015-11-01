function centeredKernelMatrix = centerKernelMatrix(kernelMatrix)
% centering the data in the feature space only using the (uncentered) Kernel-Matrix
%
% INPUT: 
%       kernelMatrix -- uncentered kernel matrix
% OUTPUT: 
%       centeredKernelMatrix -- centered kernel matrix

  % get size of kernelMatrix
  n = size(kernelMatrix, 1);

  % get mean values of each row/column
  columnMeans = mean(kernelMatrix); % NOTE: columnMeans = rowMeans because kernelMatrix is symmetric
  matrixMean = mean(columnMeans);
  disp(matrixMean);
  centeredKernelMatrix = kernelMatrix;

  for k=1:n

    centeredKernelMatrix(k,:) = centeredKernelMatrix(k,:) - columnMeans;
    centeredKernelMatrix(:,k) = centeredKernelMatrix(:,k) - columnMeans';

  end

  centeredKernelMatrix = centeredKernelMatrix + matrixMean;

end 