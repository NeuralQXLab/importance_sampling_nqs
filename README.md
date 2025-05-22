# Importance Sampling Neural Quantum States
This repository holds the code for the paper 'Importance Sampling Neural Quantum States'.

## Content of the repository
 - `conf` : some yaml config files to replicate the results in the paper using hydra
 - `data` : the data used to produce the results in the paper
 - `packages` : the code to run the experiment. It is based on a modified version of `NetKet`.
 - `notebooks` : notebooks that replicate small results in the paper, to get a better understanding of the code and play around with hyperparameters.

## To use:
First clone the repository, then install the package doing:

```bash
git clone https://github.com/NeuralQXLab/importance_sampling_nqs.git
cd importance_sampling_nqs
pip install -e .
```

You can then explore the examples stored in the `notebooks` folder, or run your own experiments using `hydra`, for example:

```bash
python main.py --config_name n2_gs is_distrib.alpha=0.5
```
## Details

The core of the importance sampling code is stored in the `advanced_drivers` package. It implements new features based on NetKet's `AbstractVariationalDriver`. Besides standard keywords, implementing Importance Sampling revolves around two new arguments:

 - `sampling_distribution` which should be a subclass of `AbstractDistribution`, also in the `advanced_drivers` package. For now, only a few are implemented :
     - $|\psi(x)|^\alpha$ is implemented under the name `OverdispersedDistribution`.
     - $|\bra{x}U\ket{\psi}|^\alpha$ where U is an operator, under the name `OverdispersedLinearOperatorDistribution`.
 -  a boolean value `auto_is` to specify wether or not to use automatic tuning of the target distribution.
## Cite

## Further Information
For further questions about what is in the repository, contact antoine.misery@polytechnique.edu.


