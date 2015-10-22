COPYRIGHT
=========

This package contains Matlab source code to perform novelty detection with KNFST as described in:

Paul Bodesheim and Alexander Freytag and Erik Rodner and Michael Kemmler and Joachim Denzler:
"Kernel Null Space Methods for Novelty Detection".
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2013.

Please cite that paper if you are using this code!

(LGPL) copyright by Paul Bodesheim and Alexander Freytag and Erik Rodner and Michael Kemmler and Joachim Denzler



CONTENT
=======

calculateKNFST.m
learn_multiClassNovelty_knfst.m
test_multiClassNovelty_knfst.m
learn_oneClassNovelty_knfst.m
test_oneClassNovelty_knfst.m
learn_oneClassNovelty_knfst_artificialClass.m
test_oneClassNovelty_knfst_artificialClass.m
README.txt  
License.txt


USAGE
=====


Multi-class novelty detection: 
  - Use the method "learn_multiClassNovelty_knfst" to learn a multi-class KNFST model and the method "test_multiClassNovelty_knfst" to compute novelty scores with the learned model.
  - Please refer to the documentations in those methods for explanations of input and output variables.

One-class classification (recommended strategy):
  - Use the method "learn_oneClassNovelty_knfst" to learn a one-class KNFST model and the method "test_oneClassNovelty_knfst" to compute novelty scores with the learned model.
  - Please refer to the documentations in those methods for explanations of input and output variables.

One-class classification (alternative strategy with artificial class):
  - Use the method "learn_oneClassNovelty_knfst_artificialClass" to learn a one-class KNFST model and the method "test_oneClassNovelty_knfst_artificialClass" to compute novelty scores with the learned model.
  - Please refer to the documentations in those methods for explanations of input and output variables.





