# Cutting-plane methods with applications in convex optimization and active learning

The subtree `project/david` contains 
* the folder `lab`;
* the folder `report`;
* the folder `writing`;
* `plan.md`; and
* `project_description.txt`

In the folder `lab` the `.py` files contain the implementation of algorithms and computations considered in this project as well as procedures used for testing purposes. The names of the `.py` files are fairly self-explanatory. For instance, `accpm.py` contains the implementation of the ACCPM algorithm. (The exception would perhaps be `config.py` which was employed to count the number of rejections made due to rejection sampling.) Testing and experimental results of these algorithms can be found in the `.ipynb` notebooks. Here, experiment notebooks have their file name start with `experiment_` while testing notebooks have their file name start with `test_`. Some of the notebooks, such as `experiment_log_regression.ipynb` have characteristics of experiments and testing, so a decision was made that seems more representative. 

The folder `report` contains a `.pdf` of the report as well as the `.tex` files of the report and slides. `report/figures` contains those images and figures that are required by `report.tex`. `slides.tex` does require any images and figures from `report/figures`. The folder `report/unused_results` contains those results that did not end up in the report. Particularly, it contains experimental results for the diabetes experiment undertaken in the report for 30 rather than 15 iterations.

The folder `writing` is intended for writings that do not fit better in another folder. It currently contains `quickstart_joining_a_github_project.tex` a guide on the basics of using Git to join a GitHub project.

`plan.md` outlines the trajectory taken by the project and `project_description.txt` outlines the project's initial goals and outcomes.