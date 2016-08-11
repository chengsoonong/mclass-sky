# David Wu - Advanced studies project (1 Semester)

## Goals:
- Understand the Analytic Center Cutting Plane Method (ACCPM).
- Use it for Active Learning on the SDSS photometric classification data (stars vs galaxies).
- Specifically: add ACCPM to the Louche and Ralaivola paper, and apply to SDSS.

## Plan:

29 July
- Set up python 3 using [anaconda](https://www.continuum.io/downloads)
- git (pull request project description) [tutorial](https://www.atlassian.com/git/tutorials/what-is-version-control)
- [jupyter](http://jupyter.org/) notebook
- sklearn tutorial [3 class iris](http://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html)

12 Aug
- Feed forward neural networks (Chapter 5.1-5.4, Bishop)
- Logistic Regression (Chapter 4.3, Bishop)
- Assignment (due 12 August): 2 marks, closing issue #103
- Assignemnt (due 18 August): 8 marks, Logistic regression notebook, closing issue #102

26 Aug
- Perceptron (Chapter 4.1.7, Bishop)
- Active Learning (Settles paper)
- Cutting Plane (Boyd, Vandenberghe lecture notes). Close issue #104

23 Sept (2 week break)
- Newton Method (Chapter 9.5, Boyd and Vandenberghe)
- Assignment (10 marks): Implement ACCPM in numpy

7 Oct
- Cutting Plane for Active Learning (Louche and Ralaivola)
- (?) Convergence of cuts (Ye)
- (?) Active Learning (Dasgupta paper)

21 Oct
- Tentative seminar date (10%)
- Assignment (10 marks): Complete experiemnt in Louche and Ralaivola with ACCPM, compare to active SVM.

4 Nov
- Tentative report date (60%)

## References

Christopher Bishop, Pattern Recognition and Machine Learning [book](http://www.springer.com/gp/book/9780387310732)

Alasdair Tran, Photometric Classification with Thompson Sampling, ANU Honours Thesis, 2015. [pdf](https://github.com/chengsoonong/mclass-sky/blob/master/projects/alasdair/thesis/tran15honours-thesis.pdf)

Burr Settles, Active Learning in Practice, 2011. [pdf](http://www.jmlr.org/proceedings/papers/v16/settles11a/settles11a.pdf)

Sanjoy Dasgupta, Two faces of Active Learning, Theoretical Computer Science, 2011. [pdf](http://cseweb.ucsd.edu/~dasgupta/papers/twoface.pdf)

Boyd, Vandenberghe, Skaf, ACCPM lecture notes [pdf](https://see.stanford.edu/materials/lsocoee364b/06-accpm_notes.pdf)

Boyd, Vandenberghe, Cutting plance lecture notes [pdf](http://web.stanford.edu/class/ee364b/lectures/localization_methods_notes.pdf)

Boyd and Vandenberghe, Convex Optimisation. [book](http://stanford.edu/~boyd/cvxbook/)

Yinyu Ye, Interior Point Algorithms, 1997
Chapter 3: Computation of Analytic Center
Section 8.1: Analytic Centers of Nested Polytopes

Ugo Louche and Liva Ralaivola, From Cutting Plane Algorithms to Compression Schemes and Active Learning, IJCNN 2015. [pdf](http://arxiv.org/pdf/1508.02986v1.pdf)
