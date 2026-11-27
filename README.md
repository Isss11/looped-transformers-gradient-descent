# looped-transformers-gradient-descent

This is the simulation code for reproducing the experiments in *Breaking Looped Transformer Multi-step Gradient Descent Assumptions for
In-context Learning* by Isaiah Sinclair, which is a final project for CS 886 at the University of Waterloo in the Fall 2026 semester.

This is a forked repository from [chengxiang/LinearTransformer](https://github.com/chengxiang/LinearTransformer), which was used to reproduce experiments for *[Transformers learn to implement preconditioned gradient descent for in-context learning](https://arxiv.org/abs/2306.00297)* by Ahn et al. and *[Linear attention is (maybe) all you need (to understand Transformer optimization)](https://arxiv.org/abs/2310.01082)* by Ahn et al.

## File Overview

### Experiments

- [`regular_experiments.ipynb`](regular_experiments.ipynb): Reproduces Figure 2 of Gatmiry et al. (2024) by training standard multi-layer (layered) and shared-weight looped linear transformers at 1, 2, 5 and 10 layers/loops, then comparing their test loss, learned P/Q matrices and how far Q is from the identity.
- [`oos_regular_experiments.ipynb`](oos_regular_experiments.ipynb): Trains looped transformers on identity-covariance data and tests them on a rotated and rescaled covariate distribution, including how test loss changes as you run 1 to 50 loops at inference.
- [`data_distributions_experiments.ipynb`](data_distributions_experiments.ipynb): Trains looped transformers separately on other data distributions (the non-Gaussian modes such as gamma, plus ReLU- and MLP-generated nonlinear data) to check whether training still converges and how the learned parameters change.
- [`gradient_dominance_experiments.ipynb`](gradient_dominance_experiments.ipynb): Checks Theorem 4.3 (gradient dominance) for looped linear transformers by recording the loss and the gradient of Q at every iteration, for both identity and non-identity population covariance (Theorem 4.3′).
- [`multi_head_experiments.ipynb`](multi_head_experiments.ipynb): Repeats the baseline looped-transformer experiment while changing the number of attention heads (1, 2, 4, 8) to see how attention width affects training and test loss.
- [`softmax_attention_experiments.ipynb`](softmax_attention_experiments.ipynb): Compares linear and softmax attention in looped transformers with 1 and 2 heads, using the same training setup and plots as the baseline.
- [`unconstrained_parameters_experiments.ipynb`](unconstrained_parameters_experiments.ipynb): Compares the paper's frozen value matrix P against a fully trained P and against a learned bias vector u, measuring test loss, Q's distance to a scaled identity, how far P moves, and whether u goes to zero.

### Helper Files

- [`linear_transformer.py`](linear_transformer.py): Defines the linear/softmax attention unit, the layered and looped transformer models, the in-context loss, and the data generators (Gaussian, gamma, ReLU, MLP and so on).
- [`experiments_helper.py`](experiments_helper.py): Holds the shared problem settings plus the functions the notebooks use to build models, run training sweeps, compute losses and distance to identity, and plot results.

## Set-up

1. Set-up a virtual environment.

```
python3 -m venv .env
```

2. Activate environment

```
. .env/bin/activate
```

3. Install requirements in the activated environment.

```
pip install -r requirements.txt
```

## Running Notebooks

After setting-up the environment, you can run any of the noteboooks in any order you wish -- I would recommend running them in VS Code (you may have to install ipykernel).
However, make sure to run the cells in the order in which they're displayed, as later cells tend to depend on earlier cells.
The only caveat for this is the cells that have a single function call that say something along the lines of `restore_results()`.
These cells are for restoring results that have been saved to a `.pth` file, and thus should not be called unless you have run and saved results previously.
By default, the notebook will use data/variables from the model that you loaded and trained in the current session if you don't run this cell, which is usually what is preferred.
