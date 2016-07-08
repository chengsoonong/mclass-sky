# An empirical study on combining active learning suggestions

## Tasks
* binary classification
* multiclass classification

Base learner: Logistic regression
* Warm start?
* polynomial kernel of degree 2

Performance:
* Accuracy
* Posterior mean balanced accuracy
* F1 score

Random sampling, figure out asymptote of performance.

Previous survey: http://www.jmlr.org/papers/volume5/baram04a/baram04a.pdf

## Active learners

Reward: improvement in performance.

* uncertainty sampling
  - entropy of probability
  - margin
* version space reduction
  - QBB margin
  - QBB KL
* (slow) Reduction of expected variance
* (slow) Reduction of expected entropy
* (maybe) information density http://burrsettles.com/pub/settles.emnlp08.pdf

## Rank aggregation
http://plato.stanford.edu/entries/social-choice/

* Pairwise majority rule
* Borda count
* geometric mean http://arxiv.org/abs/1410.4391


## Bandits
http://research.microsoft.com/en-us/um/people/sebubeck/SurveyBCB12.pdf

* kl-UCB http://arxiv.org/pdf/1210.1136.pdf
* OC-UCB http://arxiv.org/pdf/1507.07880v2.pdf
* EXP3++ http://jmlr.org/proceedings/papers/v32/seldinb14.pdf
* Thompson sampling

## Data

### UCI Data

* binary classification:
  * [ionosphere](https://archive.ics.uci.edu/ml/datasets/Ionosphere): Radar data.
  * [pima](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes): Pima Indians Diabetes.
  * [sonar](https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks)): We want to discriminate between sonar signals bounced off a metal cylinder and those bounced off a roughly cylindrical rock.
  * [wpbc](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Prognostic)): Prognostic Wisconsin breast cancer.
* multiclass:
  * [iris](https://archive.ics.uci.edu/ml/datasets/Iris): Well-known dataset from Fisher, three classes.
  * [glass](https://archive.ics.uci.edu/ml/datasets/Glass+Identification): Glass identification, seven classes.
  * [vehicle](https://archive.ics.uci.edu/ml/datasets/Statlog+(Vehicle+Silhouettes)): Classifying a given sihouette as one of four types of vehicle.
  * [wine](https://archive.ics.uci.edu/ml/datasets/Wine): Using chemical analysis to determine the origin of wines.

### SDSS

* binary classification: stars vs galaxies
* multiclass: stars vs galaxies vs quasars

## Appendix?
* Posterior balanced Accuracy derivation
* Reddening correction
* Feature selection, best kernel is poly degree 2
* Choose value of C

## Literature

* http://www.dbs.ifi.lmu.de/~tresp/papers/icml2006_actv.pdf
* http://papers.nips.cc/paper/2924-robust-design-of-biological-experiments.pdf
* http://link.springer.com/referenceworkentry/10.1007/978-3-642-04898-2_434
* http://www.jstor.org/stable/pdf/2346165.pdf
* http://ai.stanford.edu/~stong/papers/tong_thesis.pdf
