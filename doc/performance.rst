Performance Measures
====================

Here we explore various functions that evaluate the performance of a
classifier. We start by defining some notations:

-  :math:`C` is the confusion matrix, where entry :math:`C_{ij}` is the
   number of objects in class :math:`i` but have been as class
   :math:`j`,
-  :math:`m_i` is the number of objects in class :math:`i`:
   :math:`m_i = \sum_j C_{ij}`,
-  :math:`m` is the total sample size: :math:`m = \sum_i m_i`,
-  :math:`k` is the number of classes.

Naive Accuracy
--------------

One simple way to evaluate the overall performance of a classifier is to
compute the naive accuracy rate, which is simply the total fraction of
objects that have been correctly classified:

.. math::
    \text{Naive Accuracy Rate} = \dfrac{\sum_i C_{ii}}{m}

This is implemented in the function :py:func:`~mclearn.performance.naive_accuracy`.
    

Balanced Accuracy
-----------------

However, if the dataset is imbalanced, this measure would not work well.
A better approach is to use the posterior balanced accuracy. Let
:math:`A_i` be the accuracy rate of class :math:`i`:

.. math::
       A_i = \dfrac{C_{ii}}{m_i}

Before running a classifier, we know nothing of its performance, so we
can assume the accuracy rate follows a flat prior distribution. In
particular, the Beta distribution with parameters
:math:`\alpha = \beta = 1` (i.e. a uniform distriubtion) seems
appropriate here:

.. math::
       A_i \sim Beta(1, 1) \qquad \forall i

Given an accuracy rate :math:`A_i` for each class :math:`i`, the number
of correct predictions in class :math:`i` will follow a Binomial
distribution with :math:`A_i` as the probability of success:

.. math::
       \big( C_{ii} \mid A_i \big) \sim Bin\big(m_i, A_i\big)  \qquad \forall i

In Bayesian terms, this is our likelihood. Now we know that with respect
to a Binomial likelihood, the Beta distribution is conjugate to itself.
Thus the posterior distribution of :math:`A_i` will also be Beta with
parameters:

.. math::
       \big( A_i \mid C_{ii} \big) \sim Beta \big( 1 + C_{ii}, 1 + m_i - C_{ii} \big) \qquad \forall i

:py:func:`~mclearn.performance.get_beta_parameters` is
a helper function that extracts the beta paremeters form a
confusion matrix.
    

Convultion
~~~~~~~~~~

One way to define the balanced accuracy :math:`A` is to take the average
of the individual accuracy rates :math:`A_i`:

.. math::
       A = \dfrac{1}{k} \sum_i A_i

We call :math:`\big( A \mid C \big)` the posterior balanced accuracy.
One nice feature of this measure is that it's a probability distribution
(instead of a simple point estimate). This allows us to construct
confidence intervals, etc. And even though there is no closed form
solution for the density function of :math:`\big( A \mid C \big)`, we
can still compute it by performing the convolution :math:`k` times. This
is implemented in :py:func:`~mclearn.performance.convolve_betas`.

    

Expected Balanced Accuracy
~~~~~~~~~~~~~~~~~~~~~~~~~~

We have just approximated the density of the sum of :math:`k` Beta
distributions. The next step is to present our results. One measure we
can report is the expected value of the posterior balanced accuracy. This
is implemented in :py:func:`~mclearn.performance.balanced_accuracy_expected`.

    

Distribution of Balanced Accuracy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can also construct an empirical distribution for the posterior
expected accuracy. First we need to compute the pdf of the sum of beta
distributions :math:`\sum_i A_i`, given a subset :math:`x` of the
domain. See :py:func:`~mclearn.performance.beta_sum_pdf`.    

However we're interested in the average of the accuracy rates,
:math:`A = \dfrac{1}{k} \sum_i A_i = \dfrac{1}{k} A_T`. We can rewrite
the pdf of :math:`A` as:

.. math::
    F_A (a) &= \mathbb{P} (A \leq a) \\
            &= \mathbb{P}\bigg( \dfrac{1}{k} \sum_i A_i \leq a \bigg) \\
            &= \mathbb{P}\bigg( \sum_i A_i \leq ka \bigg) \\
            &= F_{A_T}(ka)

Differnetiating with respect to :math:`a`, we'd get:

.. math::
    f_A(a) &= \dfrac{\partial}{\partial a} F_A(a) \\
           &= \dfrac{\partial}{\partial a} F_{A_T}(ka) \\
           &= k \cdot f_{A_T} (ka)

See :py:func:`~mclearn.performance.beta_avg_pdf` for the implementation.
    

To make a violin plot of the posterior balanced accuracy, we need to run
a Monte Carlo simulation, which requires us to have the inverse cdf of
:math:`A`. :py:func:`~mclearn.performance.beta_sum_cdf`,
:py:func:`~mclearn.performance.beta_avg_cdf`, and
:py:func:`~mclearn.performance.beta_avg_inv_cdf` are used to approximate
the integral of the pdf using the trapezium rule.
    

Recall
------

We can also compute the recall of each class. The recall of class
:math:`i` is defined as:

.. math::
    \text{Recall}_i = \dfrac{C_{ii}}{\sum_j C_{ij}}

Intuitively, the recall measures a classifier's ability to find all the
positive samples (and hence minimising the number of false negatives).
See :py:func:`~mclearn.performance.recall`.
    

Precision
---------

Another useful measure is the precision. The precision of class
:math:`i` is defined as:

.. math::
    \text{Precision}_i = \dfrac{C_{ii}}{\sum_j C_{ji}}

Intuitively, the precision measures a classifier's ability to minimise
the number of false positives. See :py:func:`~mclearn.performance.precision`.   
