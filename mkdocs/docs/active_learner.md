Active Learning
===============

(placeholder)

Main Active Learning Routine
----------------------------

(placeholder)

Heuristics
----------

(placeholder)

### Random Benchmark

(placeholder)

### Uncertainty Sampling

(placeholder)

### Query by Bagging

The Kullback-Leibler divergence of $Q$ from $P$ is defined as

$$D_{\mathrm{KL}}(P\|Q) = \sum_i P(i) \, \ln\frac{P(i)}{Q(i)}.$$

This KL divergence measures the amount of information lost when $Q$ is
used to approximate $P$. In the active learning context, $Q$ is the
average prediction probability of the committee, while $P$ is the
prediction of a particular committee member.

