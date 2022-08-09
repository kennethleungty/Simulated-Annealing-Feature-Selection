# Feature Selection with Simulated Annealing in Python, Clearly Explained
### Concepts and Python implementation of the global search algorithm to select the best features for machine learning
___
Link to Medium article: *Coming Soon*
___
### Context
Feature selection is a key step in machine learning as it boosts computational efficiency and predictive performance by keeping only the most relevant predictors.

Beyond the popular supervised feature selection classes like filter and wrapper methods, global search methods like  simulated annealing are also powerful techniques at our disposal.

In this project, we delve into the theory and application of simulated annealing for feature selection.
___
### Project Structure
- `data`: Titanic dataset (raw and processed)
- `images`: Set of images and visualizations used to demonstrate algorithm
- `notebooks`: Jupyter notebooks for the different steps of the project i.e. data pre-processing, baseline modeling, and running of feature selection with simulated annealing algorithm
- `results`: CSV files of the output from algorithm runs
- `src`: Python scripts for simulated annealing algorithm for feature selection
    - `main.py`: Main script containing algorithm. In CLI, `cd` into `src` folder, then execute `python main.py`
    - `utils.py`: Utils script containing ML model function (i.e. random forest classifier)
___
### References
- https://researchgate.net/publication/227061666_Computing_the_Initial_Temperature_of_Simulated_Annealing
- https://search.r-project.org/CRAN/refmans/caret/html/safs.html
- https://iopscience.iop.org/article/10.1088/1742-6596/1752/1/012030/pdf
- https://topepo.github.io/caret/feature-selection-using-simulated-annealing.html
- https://www.slideshare.net/kaalnath/simulated-annealingppt
- https://santhoshhari.github.io/simulated_annealing/
- https://www.feat.engineering/simulated-annealing.html
- https://towardsdatascience.com/optimization-techniques-simulated-annealing-d6a4785a1de7
- https://www.sciencedirect.com/science/article/abs/pii/S0377221704005892
- https://www.youtube.com/watch?v=Dp1irQX-c0Q&ab_channel=Udacity 
- https://link.springer.com/article/10.1007/s10878-020-00607-y 
- https://arxiv.org/pdf/1906.01504.pdf 