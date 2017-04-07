Run
```{python}
    photometric_redshift.py GP data.csv
```
to fit model and compute $R^2$ score.

Replace `GP` with `SGD` or `const` to run Stochastic Gradient Descent regression or a dummy (constant) predictor, respective.y

`-p` plots and `-t` tests the $R^2$ calculation.
