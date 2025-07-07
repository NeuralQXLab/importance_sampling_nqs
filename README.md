# Importance Sampling for Neural Quantum States
This repository holds the code for the paper 'Importance Sampling Neural Quantum States'.

## Content of the repository
 - `conf` : some yaml config files to replicate the results in the paper using hydra
 - `data` : the data used to produce the results in the paper
 - `packages` : the code to run the experiment. It is based on a modified version of `NetKet`. `advanced_drivers` modifies the `AbstractVariationalDriver` to incorporate additional features such as SPRING, MinSR, and in our case importance sampling. `ptvmc` and `deepnets` hold some model classes, such as the CNN and the Vision transformer (ViT). In `grad_sample`, you'll mostly find the training pipelines where you can load everything from a single yaml config file using hydra to run several experiments. `netket_pro` provides a few useful extra modifications to NetKet classes or Jax functions.
 - `notebooks` : notebooks that replicate small results in the paper, to get a better understanding of the code and play around with hyperparameters.

## To use:
First clone the repository, then install the package doing:

```bash
git clone https://github.com/NeuralQXLab/importance_sampling_nqs.git
cd importance_sampling_nqs
uv sync
```

By using [uv](https://docs.astral.sh/uv/getting-started/installation/) you will get the EXACT same dependencies that we used and are guaranteed to work, which are recorded into the uv.lock file.
You can also use pip, but we strongly discourage you from doing so. 
To use pip instead, do
```bash
pip install -e .
```

You can then explore the examples stored in the `notebooks` folder, or run your own experiments using `hydra`.
Here are a list of commands to replicate the main results of the paper:
- Nitrogen ground state. Here we set the sampling distribution to be $|\psi|^{\frac12}$ 
```bash
uv run main.py --config-name n2_gs is_distrib.alpha=0.5
```
- Lithium oxide ground state, with adaptive tuning starting at $|\psi|$ for more stability
```bash
uv run main.py --config-name li2o_gs is_distrib.alpha=1.0 auto_is=true
```
- J1J2 with a ViT, on the 6x6 square lattice:
```bash
uv run  main.py --config-name vit_j1j2 is_distrib.alpha=2.0 auto_is=true model.L=6
```
- Signal-to-noise ratio study on the nitrogen molecule:
```bash
uv run main.py --config-name snr_analysis model=nitrogen ansatz=nnbf ansatz.hidden_units=32
```
- Infidelity minimization:
```bash
uv run main.py --config-name infidelity_tfi
```

### Note: no uv
If you do not want to use uv, then simply replace `uv run` with `python` in the commands above.


## Details

The core of the importance sampling code is stored in the `advanced_drivers` package. It implements new features based on NetKet's `AbstractVariationalDriver`. Besides standard keywords, implementing Importance Sampling revolves around two new arguments:

 - `sampling_distribution` which should be a subclass of `AbstractDistribution`, also in the `advanced_drivers` package. For now, only a few are implemented :
     - $|\psi(x)|^\alpha$ is implemented under the name `OverdispersedDistribution`.
     - $|\bra{x}U\ket{\psi}|^\alpha$ where U is an operator, under the name `OverdispersedLinearOperatorDistribution`.
 -  a boolean value `auto_is` to specify wether or not to use automatic tuning of the target distribution.

 For a deeper dive in the code, you can look at `./packages/advanced_drivers/_src/driver/ngd/driver_abstract_ngd.py`, `./packages/advanced_drivers/_src/driver/ngd/sr_srt_common.py` which hold most of the importance sampling machinery. The weights are computed in the `_prepare_derivatives` function in the driver. The reweighting is then done in the function `_prepare_input` in `sr_srt_common.py`. The function for adaptive tuning is `_compute_snr_derivative` and can in `sr.py` in the same folder.

## Cite

## Further Information
For further questions about what is in the repository, contact antoine.misery@polytechnique.edu.


