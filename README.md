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
cd importance_sampling_nqs
pip install -e .
```

You can then explore the examples stored in the `notebooks` folder, or run your own experiments using `hydra`, for example:

```bash
python main.py --config_name n2_gs is_distrib.alpha=0.5
```

## Cite

## Further Information
For further questions about what is in the repository, contact antoine.misery@polytechnique.edu.


