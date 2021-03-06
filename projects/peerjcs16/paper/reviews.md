# Reviews

## Reviewer 1 (Anonymous)

### Basic reporting

Generally speaking, the article is well written and pleasant to read.

A few comments on the structure and on the reporting:

- It might be more natural to place the paragraph on information-density weighting (lines 313-322) in Section 2.1.
  - **We’ve moved this paragraph to the end of Section 2.1.**
- Using rank aggregation to combine multiple AL strategies has been proposed before [1], albeit in the context of active multi-task learning. I believe this should be acknowledged.
  - **A reference has been added to the end of Section 1.**
- The above reference can be found, e.g., in Burr Settles' book [2], which might also be a useful addition to the references.
  - **We have added a reference to this book just before Section 2.1.**
- Table 4: "Max Performance" -> the metric is not specified.
  - **The metric (MPBA) is now specified.**
- Figures 4 and 5: It would be helpful to indicate which quantiles are represented by the boxes and the whiskers.
  - **We have added this to the caption of (now) Figures 6 and 7.**

[1] R. Reichart et al., Multi-Task Active Learning for Linguistic Annotations, ACL 2008.
[2] B. Settles, Active Learning, 2012, Morgan & Claypool

### Experimental design

The research question that the authors seek to address is important and meaningful. Their approach---analyzing a large number of methods on a large number of datasets---is a priori interesting. The experimental methods are well described, and the code is easy to read.

However, I believe that there are important issues in the experimental design, such that it is not possible to gain a clear, objective understanding of the performance of the AL strategies. Hence, in its current state, the paper fails to properly address the research question.

1. The strength (eq. 25) is upper-bounded by 1, assuming that the discrete integrals are positive. However, looking at Fig. 4 and 5, we see a number of instances where the magnitude of the strength is much larger than 1, positively or negatively. This is due to the fact that $m(\max)$ in (25) is the performance achieved using all labeled data, which, in some cases, is *not* the best possible performance. In these cases, the discrete integral can be 0 or negative, and the strength might not even be well-defined. This seriously impacts the interpretability of the results. The following solutions might be explored:

    - $m(\max)$ could be redefined as the best performance in the given run (such that by construction the discrete integrals will be positive), or
    - multiple runs could be averaged before computing the strength, in order to decrease the variability in the performance, or
    - the authors might rely less on strength measurements and more on the learning curves (Fig. 6 and 7), which should then be significantly improved, e.g., by averaging more runs (>=50) and by showing more strategies, or
    - other performance measures might be considered, as they might prove to be less sensitive to the "nonpositive integral" problem.

    **You’re right. We decided to follow your first suggestion and redefined the strength asthe best performance in a given trial, where we take the higher of**

    **1)   The performance achieved by using allthe labeled examples in the training set.**

    **2)   The maximum value of the learning curvesof all the methods.**

    **This ensures that the strength is upper-bounded by 1. We have re-drawnall the figures to include the updated strength. We have added a footnote tothe definition of $m(\max)$ under the strength definition (Equation 25). In thelearning curve figures, the horizontal dashed line now indicates the average performancefrom using all training examples (which can be lower than parts of certainactive learning curves, supporting the claim that simply using more datadoesn’t necessarily improve performance).**

2. There is no meaningful comparison with prior work. For example, I suggest considering the datasets & classifier of [3] and including the results of this paper in the plots, so that it is possible to get a clear comparison between the two.

    **[3] presents the COMB algorithm, which uses the classification entropy**
    **maximization score. This score has only been shown to work with binary datasets**
    **and support vector machines. Our work focuses more on algorithms that work with**
    **both binary and multiclass classification. We also use logistic regression**
    **(rather than the SVM used in [3]), which can be trained to produce probability**
    **estimates directly.**

    **We have however expanded the last paragraph in Section 3.1, where we**
    **discussed the similarities and differences between our work and prior work such**
    **as [3] and [4].**

3. The results seem to depend a lot on the choice of hyperparameters. These dependencies need to be explored more systematically. A few examples:

    - Thompson sampling: the authors state that it favors certain heuristics a lot more than others. But this is likely going to be very different if $\tau^2_r$ and the initial $\mu_r$ are set differently.

      - **We run some additional experiments to investigate how changing \mu,**
        **\sigma^2 and \tau^2 affects the results. See Figure 5 and 11. Changing these**
        **values does not have much effect on the final performance, while it still**
        **favors certain heuristics over others. This observation has been added to the**
        **end of Section 4.**

    - Poor performance of information-density weighted strategies: this obviously depends a lot on $\gamma$.

      - **We run some more experiments to test the effect of \gamma on two of the datasets,**
        **glass and ionosphere. See Figure 4. The value of \gamma has little effect on**
        **performance of the glass dataset, while for the ionosphere dataset, using the 90th or higher percentile**
        **of the pairwise distances between the data points seems to work better (but**
        **they are still not as good as the heuristics without the information density**
        **weighting).**

    - Is there an effect of the size of the candidate pool?

      - **Originally,we restricted the candidate pool size to 300 because we included the heuristics that involve minimizingthe variance or maximizing the classifier certainty of the model, both of whichare computationally intensive. We later excluded these heuristics from ourstudy, so now there is no need to have this restriction. Now in ourexperiments, we have assign a score to all examples in the unlabelled pool.**

        **In addition, we have also tested the effect of the unlabelled pool on the resultSee Figure 12. Overall, having a smaller pool does reduce the performance. This result has been added to Section 6.**

    - Regularization strength in Eq. 22: the choice of $C = 1000$ seems somewhat arbitrary and needs to be better explained.

      - **We**
        **have re-run the experiment so that the C value is chosen with a grid search,**
        **which finds the best value between 10^-6 and 10^6.**

[3] Y. Baram et al., Online Choice of Active Learning Algorithms, 2004, JMLR.

**[4] Hsu, W.-N. and Lin, H.-T. (2015). Active learning by learning. In AAAI 15**

### Validity of the findings

1. The authors claim that "with the small and medium-sized datasets, the effect of active-learning is minimal". Do the authors suggest that AL should only be used if one has access to a large unlabeled pool? Or that the number of AL iterations has to be sufficiently large? I believe this might be more of a measurement problem: with small datasets, there are fewer iterations, the test set is smaller, and the measured strength is noisier---in conclusion, this might just be an artefact of the experimental procedure. In any case. this should be clarified, perhaps by running additional experiences (e.g., by looking at a small sample of a large dataset, sample whose size matches that of the small datasets).

   **We ran additional experiments and tested the effect of the data size byartificially reducing either the unlabeled pool size or both the unlabeled poolsize and test size.**

   **See Figure 12. It is indeed true that having a smaller unlabeled poolsize reduces the accuracy and having a smaller test size further deceases it. This observation is now added to the Section 6.**

2. Line 353: "there is no statistical difference between the corresponding heuristic/combiner and passive learning". Clearly, looking e.g. at the median in Fig. 4., there seems to be a difference. Do the authors mean that the difference is not statistically significant?

   **Yes, we mean the difference is statistically small. We’ve rewritten the**
   **results section and made it clearer.**

### Comments for the Author

Small comments:

- You refer multiple times to the "optimal heuristic". I believe "best heuristic" or "heuristic that works best" is more appropriate.

  - **Fixed**

- Line 59: "test example" -> maybe "candidate example"?

  - **Fixed**

- Eq. 5: why the $x$ in $r(x, h)$? The $x$ on the RHS is in a different scope (it is argmax'ed over). Maybe it should be $r(\mathcal{U}, h)$ instead.

  - **We agree. We’ve changed it to $r(\mathcal{U}; h)$**

- Sec 3.1. A general reference to multi-armed bandits would be welcome after the first sentence. Furthermore, I am not sure that the typical objective of MABs is to "minimize the number of pulls" (lines 132-133).

  - **We have added a reference to Berry & Fristedt (1985)’s**
    **monograph on bandit problems. We’ve removed the phrase “minimize the number of**
    **pulls”. The objective of MABs is indeed more about maximising the rewards in**
    **the long run.**

- Figure 1: why the dotted line? What is the difference with plain lines?

  - **It should be a solid line. Fixed.**

- Eq. 12: the way the function is defined implies that the bandit is able to see a reward for every arm, but that is not the case, I presume? I think the domain of the function should be revised.

  - **You’re right. We’ve updated the domain (now in Eq. 15) to make it**
    **clearer.**

- Eq 13: why $| v_r$ on the LHS?

  - **This notation makes it clear that we are fixing (i.e. conditioning on) v_i,**
    **since v_i is a random variable. When conditioning on v_i, W_i follows a normal**
    **distribution with mean v_i.**

- Algorithm 2, last line: "update the parameters of $B$" -> should be small case $b$.

  - **Fixed**

- Line 201: "consider the KL-divergence" -> between which distributions?

  - **Between the distribution of the current estimated reward and that of the upper**
    **bound. We’ve clarified this line.**

- Line 203: "Algorithm" -> Algorithms

  - **Fixed**

- Line 226: "we could only use one suggestion from one particular heuristic" -> from what I understand, this is not the case for EXP4.

  - **You’re right. We have removed this line. The main advantage of the**
    **social choice theory would then be that it doesn’t require a reward scheme.**

- Line 236: $k(s(x; h))$ -> s(x:h) is a score (i.e., a real number), this is inconsistent with the definition of $k$ in Eq. 20.

  - **It’s a bit tricky to get the notation right. Since the concept itself is**
    **fairly simple (converting scores into a ranking), we have simplified the**
    **notation and removed Eq. 20. The example in Table 2 should be clear enough to**
    **see how k works.**

- Line 256: Why is it necessary to rescale the ranks? I don't see why this would change the aggregation.

  - **You’re right. It doesn’t affect the aggregation. Rescaling is just a**
    **technical condition needed in the proof of the geometric rank. See [4]**

    **[4]: Bedo, J. and Ong, C. S. (2016). Multivariate spearman’s ρ for**
    **aggregating ranks using copulas. Journal of Machine Learning Research**

- Line 292: "with an L2 loss" -> I suspect you mean "with an L2 regularizer".

  - **Yes. Fixed**

- Eq. 22: $w^T \theta$ should probably be $\theta^T \theta$.

  - **Fixed**

- Line 316: "vetor" -> vector

  - **Fixed**

- Line 393: "Thompson sampling requires" -> assumes

  - **Fixed**

- Line 394 "This can be hard to satisfy" -> This assumption is unrealistic.

  - **Fixed**

## Reviewer 2 (Anonymous)

### Basic reporting

- The introduction should be improved, highlighting better the motivations and the main contributions of the work.

    - **We have rewritten the final part of the Introduction to highlight our**
        **contributions better.**

- Generally speaking, the writing of the paper is good.

- In the session “related work”, the authors should explain better the reasons to only analyze and describe the Uncertainty Sampling and Version Space Reduction strategies. The authors wrote that the other heuristics (query strategies) are very computationally expensive, but they should explain further this situation.

    - **We have further explained the reasons why we excluded other query**
        **strategies in the paragraph just before Section 2.1.**

- “Out of the five heuristics discussed, how do we know which one is the most optimal one?” → This question is quite interesting, but I think that the author should consider to rewrite it. According to the Non-Free-lunch theorem, that can be applied to active learning too, there is not a method that performs the best for all possible problems. Maybe, the question should be focused to encounter a method with a “optimal” performance for a particular set of problems.

    - **We rephrased this to “…which one should we use in practice when we**
        **would like to apply active learning to a problem?”**

- “There have been some attempts to combine active learning suggestions in the literature. Baram et al. (2004) used the EXP4 multi-armed bandit algorithm to automate the selection process. Hsu and Lin (2015) studied an improved version, EXP4.P, along with importance weighting to estimate the rewards using only the training set. This paper empirically compares the following four algorithms: Thompson sampling, OC-UCB, kl-UCB, and EXP3++.”

    - The authors should not restrict the state-of-the-art description to the single-label classification problem. For instance, in the following reference was proposed a method to average several rankings and obtain a consensus for selecting the next query in the context of multi-label learning.
    - Oscar Reyes, Carlos Morell, Sebastián Ventura, Effective active learning strategy for multi-label learning, Neurocomputing, Volume 273, 2018, Pages 494-508.

    **This is a definitely an interesting area to investigate, but we think**
    **that this is beyond the scope of this paper. We have however mention this as a**
    **limitation of our methods in the conclusion and added the above reference.**

- The authors should improve the explanation of the proposed method.

    - **Do you have any particular part in mind where the explanation is not**
        **clear?**

- Does the proposed method fulfill the Condorcet criterion?

    - **Only the Schulze method fulfill the Condorcet criterion. Borda count and**
        **geometric mean do not.**

### Experimental design
- It is not explained how the authors obtained probabilities from the logistic regression base classifier.
  - **As shown in Eq 2, logistic regression directly produces the probability**
    **of being in the positive class for binary datasets. For multiclass**
    **classification, we use the One-vs-Rest strategy, where for every class, we build a binary**
    **classifier that determines whether a particular example belongs to that class**
    **or not. This is explained in Section 4.**
- What were the criteria to select the 11 datasets used in the experimental study?. On the other hand, the authors should increase the number of datasets in order to make the study more reliable.
  Validity of the findings
  - **These datasets were chosen such that we have an equal number of binary and multiclass datasets, and a mixture of small and large datasets. We think thatthese 12 datasets contain enough variety to make useful conclusions andincreasing the number of datasets would only increase the training time and computationalresources.**
- My main concern about this work is that the authors assumed that only one sample is selected in every active learning iteration. However, nowadays this scenario is much less used since the selection of several samples at the same time is preferred in the majority of real-world applications. To select a set of queries in each active learning iteration you can follow either a myopic active learning strategy or a batch-mode active learning one. There are main differences between these two kinds of strategies. The authors should explain this situation in the paper, and portrait some tracks explaining how to adapt the proposed method to this type of scenarios.
  - **Batch active learning is certainly an interesting area to explore. We**
    **have included a discussion of this in the Conclusion.**
- The discussion and conclusions are well stated, they are limited to the results obtained.

### Comments for the Author
Generally speaking, I consider that this work is interesting, but it should be significantly improved before it could be considered for publication. I suggest that authors review the work attending the comments made by the reviewer.
