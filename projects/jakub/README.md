# Contents

 - `gaussian_processes/`: A summary of Gaussian Process regression. Compile with Xelatex.

 - `issue_145_proof/`: Proof that linear regression (with linear features) does not benefit from extra features that are the differences between other features. Compile with Xelatex.

 - `kernel_density/`: Jupyter notebook with a kernel density estimator for the photometric data.

 - `lat_lng_z_plot/`: Jupyter notebook plotting the redshift against the right ascention and declination.

 - `learning_curve/`: Code to generate passive learning curves. Call `python learning_curve.py PATH_TO_SDSS_DATA` to compute, then `python plot.py PATH_TO_DATA` to plot.

 - `papers.txt`: List of papers I want to read.

 - `redshift_regression/`: Photometric redshift regressions. Implemented GP regression, SGD (linear) regression with kernel approximation, SGD (linear) regression with linear features, and a dummy (constant) regressor.

 - `redshift.tex`: A summary of the uses of redshift. Compile with Xelatex.

 - `thesis/`: Start of the thesis, plus miscellaneous writing. Compile all .tex files with Xelatex.

 # Instructions

 To generate a .h5 file from a .csv (or .csv.gz) file, use the splitter:

 python splitter.py --to-hdf5 input.csv output.h5
