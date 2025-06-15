# Amortized Variational Transdimensional Inference

The expressiveness of flow-based models combined with stochastic variational inference (SVI) has, in recent years, expanded the application of optimization-based Bayesian inference to include problems with complex data relationships. However, until now, SVI using flow-based models has been limited to problems of fixed dimension. We introduce CoSMIC, normalizing flows (COntextually-Specified Masking for Identity-mapped Components), an extension to neural autoregressive conditional normalizing flow architectures that enables using a single amortized variational density for inference over a transdimensional target distribution. We propose a combined stochastic variational transdimensional inference (VTI) approach to training CoSMIC flows using techniques from Bayesian optimization and Monte Carlo gradient estimation. Numerical experiments demonstrate the performance of VTI on challenging problems that scale to high-cardinality model spaces.

[ArXiV link](https://arxiv.org/abs/2506.04749)

## Installation

```bash
curl -sSL https://install.python-poetry.org | python3 -

poetry install  # to just run, or
poetry install --with dev  # if you want to debug things
```

### Slurm configuration

For slurm configuration with project numbers, create an .env file

```text
SLURM_ACCOUNT=YOURPROJECTNUMBER
```

### Recommended HPC config

#### Marimo

[Marimo](https://marimo.io/) is a no-frills notebook library.
It is automatically installed by `pypoetry`.

#### Dependencies

module add python/3.12.3 tmux texlive/2022

#### Disable progress bars for third party libraries
```bash
export TQDM_DISABLE=1
```

## Running examples locally

Figure 2:

```bash
poetry run python vti/examples/local/local_robustvs_fig2_float32.py
```

Figure 3 data (example dgp seed 1000)
```bash
poetry run python vti/examples/local/local_dagmlpnobias_10node_march2025_float32.py 1000
```

## Running examples using Marimo/Slurm/Submitit

Figure 4:
```bash
poetry run marimo edit robust_vs_comparison_surrogate_sfecat_sfemade.py
```
