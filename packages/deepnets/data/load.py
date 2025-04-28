import h5py
import numpy as np
import glob
import netket as nk
import flax
from deepnets.net import ViT2D
import re
from collections.abc import Sequence


def get_h5_paths(dir: str):
    """
    Return list of paths to all .h5 files in dir
    """
    return glob.glob(dir + "*.h5")


def get_matching_h5(dir: str, match_str: str):
    """
    Return list of paths to all .h5 files in dir containing match_str
    """
    return glob.glob(f"{dir}*{match_str}*.h5")


def get_matching(dir: str, match_str: str):
    """
    Return list of paths to all files in dir containing match_str
    """
    return glob.glob(f"{dir}*{match_str}*")


def get_int(filename: str, prefix: str) -> int:
    with open(filename) as file:
        contents = file.read()
        match = re.search(rf"{prefix} (\d+)", contents)
        if match:
            number = int(match.group(1))
            return number
        else:
            print("Number not found")


def get_parameters(h5_file: h5py.File, groups: Sequence):
    """
    Read the attributes in groups from the h5_file into a dictionary
    """
    out = {}
    for group in groups:
        for key in h5_file[group].attrs.keys():
            out[key] = h5_file[group].attrs[key]

    return out


def get_mpack_paths(dir: str):
    """
    Return list of paths to all .mpack files in dir, netket variational states are saved as .mpack at the end of optimization
    """
    return glob.glob(dir + "*.mpack")


def load_hyperparamsearch(h5_files: list, hyperparam_dict: dict, n_runs=1):
    """
    Load in data from a hyperparameter search
    Input: h5_files: list of h5 file paths
           hyperparam_dict: dictionary indexed by ["hyperparam_name": hyperparam_array], with
                            hyperparam_array an array of the values for that hyperparameter
           n_runs: number of runs with the same hyperparameters
    Output: energies,times
            energies = [hyperparam1,hyperparam2,...,hyperparamN,run,iteration] array of energies during optimization
            times = [hyperparam1,hyperparam2,...,hyperparamN,run] array of CPU times for each run

    The size of the final dimension of the energies array is determined by the maximum number of iterations of all runs
    """
    hyperparam_shape = []
    for key, value in hyperparam_dict.items():
        hyperparam_shape.append(len(value))
    # print(hyperparam_shape)

    # Get maximum number of iterations from files
    iters = []
    for file in h5_files:
        with h5py.File(file, "r") as f:
            iters.append(f["Optimization"].attrs["Iterations"])
    n_iters = max(iters)

    run_counter = np.zeros(hyperparam_shape, dtype=int)
    energies = np.full(hyperparam_shape + [n_runs, n_iters], float("nan"), dtype=float)
    times = np.full(hyperparam_shape + [n_runs], float("nan"), dtype=float)
    for file in h5_files:
        # print(f"Current energies\n{energies}")
        # print(f"For file {file}")
        with h5py.File(file, "r") as f:
            hyperparam_indices = []
            for key, arr in hyperparam_dict.items():
                hyperparam_indices.append(
                    np.where(arr == f["Network"].attrs[key])[0][0]
                )
            hyperparam_indices = tuple(hyperparam_indices)
            # print(f"Hyper param indices: {hyperparam_indices}")
            # print(hyperparam_indices)
            run_index = run_counter[hyperparam_indices]
            # print(f"Run index: {run_index}")
            run_counter[hyperparam_indices] += 1
            energy_index = hyperparam_indices + (run_index, ...)
            energies[energy_index] = f["Optimization/Energy"][...]
            time_index = hyperparam_indices + (run_index,)
            times[time_index] = f["Optimization"].attrs["CPU time"]

    return energies, times


def load_optsearch(h5_files: list, opt_dict: dict, n_runs=1):
    """
    Load in data from a hyperparameter search
    Input: h5_files: list of h5 file paths
           opt_dict: dictionary indexed by ["opt_name": iot_array], with
                            opt_array an array of the values for that optimization parameter
           n_runs: number of runs with the same hyperparameters
    Output: energies,times
            energies = [opt1,opt2,...,optN,run,iteration] array of energies during optimization
            times = [opt1,opt2,...,optN,run] array of CPU times for each run
    Different from load_hyperparamsearch as attribute loaded from f["Optimization"].attrs["opt_name"]

    The size of the final dimension of the energies array is determined by the maximum number of iterations of all runs
    """
    opt_shape = []
    for key, value in opt_dict.items():
        opt_shape.append(len(value))

    # Get maximum number of iterations from files
    iters = []
    for file in h5_files:
        with h5py.File(file, "r") as f:
            iters.append(f["Optimization"].attrs["Iterations"])
    n_iters = max(iters)

    run_counter = np.zeros(opt_shape, dtype=int)
    energies = np.full(opt_shape + [n_runs, n_iters], float("nan"), dtype=float)
    times = np.full(opt_shape + [n_runs], float("nan"), dtype=float)
    for file in h5_files:
        with h5py.File(file, "r") as f:
            hyperparam_indices = []
            for key, arr in opt_dict.items():
                hyperparam_indices.append(
                    np.where(arr == f["Optimization"].attrs[key])[0][0]
                )
            hyperparam_indices = tuple(hyperparam_indices)
            run_index = run_counter[hyperparam_indices]
            run_counter[hyperparam_indices] += 1
            energy_index = hyperparam_indices + (run_index, ...)
            energies[energy_index] = f["Optimization/Energy"][...]
            time_index = hyperparam_indices + (run_index,)
            times[time_index] = f["Optimization"].attrs["CPU time"]

    return energies, times


def read_from_str(
    params: dict, key: str, match_str: str, return_type=float, end_str=","
):
    """
    Read the value in the string at params[key] = string, which comes immediately after match_str
    """
    value = params[key]
    start_index = value.index(match_str) + len(match_str)
    end_index = start_index + value[start_index:].index(end_str)
    number = return_type(value[start_index:end_index])
    return number


def network_helper(params: dict, system):
    if params["Network/name"] == "ViT2D":
        return ViT2D(
            depth=int(params["Network/depth"]),
            d_model=int(params["Network/d_model"]),
            heads=int(params["Network/heads"]),
            linear_patch_size=int(params["Network/linear_patch_size"]),
            output_head=params["Network/output_head_name"],
            system=system,
        )

    # TODO, add others


def system_helper(params: dict):
    if params["System/name"] == "Shastry-Sutherland":
        return eval(
            f"Shastry_Sutherland(L={params['System/L']},J={params['System/J']})"
        )

    # TODO, add others


def load_vstate(mpack_name: str, network, sampler) -> nk.vqs.VariationalState:
    var_state = nk.vqs.MCState(sampler, model=network)
    with open(mpack_name, "rb") as f:
        variables = flax.serialization.from_bytes(var_state.variables, f.read())
    var_state.parameters = variables
    return var_state


def get_indices(param_seq: Sequence, key: str, match_str: str):
    """
    Get all i for which params_seq[i][key] == match_str
    """
    return [i for i, param in enumerate(param_seq) if param[key] == match_str]


def sort_indices(param_seq: Sequence, sort_key: str):
    """
    Return a list of the indices of param_seq sorted according to the value of param_seq[i][sort_key]
    """
    return [
        i
        for i, _ in sorted(
            enumerate([p[sort_key] for p in param_seq]), key=lambda x: x[1]
        )
    ]
