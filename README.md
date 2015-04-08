# mclass-sky

Large Scale Classification of Astronomical Objects

This project involves developing an open-world multi-class classifier
for the [SkyMapper Southern Sky Survey](http://rsaa.anu.edu.au/research/projects/skymapper-southern-sky-survey).

## Contents


### Implementation Details
* <a href="http://nbviewer.ipython.org/github/alasdairtran/mclass-sky/blob/master/data_processing.ipynb" target="_blank">Data Processing</a>: Some useful procedures (e.g. normalisation, splitting data into training and test set) that we might want to call before feeding the data to the classifier.
* <a href="http://nbviewer.ipython.org/github/alasdairtran/mclass-sky/blob/master/performance_measures.ipynb" target="_blank">Performance Measures</a>: This notebook explains the implementation of various performance measures, including the posterior balanced accuracy.

### SDSS Classification
* <a href="http://nbviewer.ipython.org/github/alasdairtran/mclass-sky/blob/master/sdss_about_the_data.ipynb" target="_blank">SDSS - About the Data</a>: Some information about the SDSS dataset and how to obtain them.
* <a href="http://nbviewer.ipython.org/github/alasdairtran/mclass-sky/blob/master/sdss_classification.ipynb" target="_blank">SDSS - Classification</a>: We apply some of the standard classifiers on the SDSS dataset.
* <a href="http://nbviewer.ipython.org/github/alasdairtran/mclass-sky/blob/master/sdss_dim_reduction.ipynb" target="_blank">SDSS - Dimensionality Reduction</a>: We reduce the
data down to 2 features to visualise the classes.
* <a href="http://nbviewer.ipython.org/github/alasdairtran/mclass-sky/blob/master/sdss_active_learning.ipynb" target="_blank">SDSS - Active Learning</a>


### Datasets
The datasets are too big for GitHub. They can be downloaded from
[my Dropbox folder](https://www.dropbox.com/sh/yrhifi5s1qikfh8/AAAZg6BouEalG1yjbUJK4gvea?dl=0).