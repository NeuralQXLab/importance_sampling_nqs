import warnings


import jax

from netket.utils import mpi
from netket.utils.config_flags import config
from netket.jax.sharding import (
    extract_replicated,
    gather,
    distribute_to_devices_along_axis,
)

from netket.sampler import MetropolisSamplerState


# We must support serialization of the sampler state, which is a bit tricky
# because we must support multi-GPU, sharding or MPI.
def serialize_MetropolisSamplerState(sampler_state):
    # First serialize the local data with the default scheme
    state_dict = MetropolisSamplerState._to_flax_state_dict(
        MetropolisSamplerState._pytree__static_fields, sampler_state
    )

    # then post-process the data to gather the data to all devices
    if config.netket_experimental_sharding:
        # For sharding, we simply gather and unreplicate the data
        for prop in ["σ", "n_accepted_proc"]:
            x = state_dict.get(prop, None)
            if x is not None and isinstance(x, jax.Array) and len(x.devices()) > 1:
                state_dict[prop] = gather(x)
        state_dict = extract_replicated(state_dict)
        # distributed == -1 means sharding
        state_dict["distributed"] = -1
    else:
        # For MPI, we gather the data to all processes
        # here the rng is not global, so we must gather it as well.
        for prop in ["σ", "n_accepted_proc", "rng"]:
            x = state_dict.get(prop, None)
            if x is not None:
                x = mpi.mpi_allgather(x)
                y = x.reshape(-1, *x.shape[2:])
                state_dict[prop] = y

        rng = state_dict.get("rng", None)
        if rng is not None:
            if rng.ndim > 1:
                rng

        state_dict["distributed"] = mpi.n_nodes
    return state_dict


def deserialize_MetropolisSamplerState(sampler_state, state_dict):
    # backward compatibility
    if "distributed" not in state_dict:
        state_dict["distributed"] = 1

    if config.netket_experimental_sharding:
        if state_dict["distributed"] != -1:
            warnings.warn(
                "Deserializing a MetropolisSamplerState with sharding, but the current setup is not sharded. This might lead to unexpected behavior.",
                category=UserWarning,
                stacklevel=2,
            )
        del state_dict["distributed"]

        for prop in ["σ", "n_accepted_proc"]:
            x = state_dict[prop]
            if x is not None:
                state_dict[prop] = distribute_to_devices_along_axis(x)
    else:
        n_chains = sampler_state.σ.shape[0] * mpi.n_nodes
        if state_dict["distributed"] != mpi.n_nodes:
            n_chains_serialized = state_dict["σ"].shape[0]
            if n_chains_serialized == n_chains:
                warnings.warn(
                    f"Deserializing a MetropolisSamplerState saved with {state_dict['distributed']} MPI nodes."
                    "Same number of chains detected: using same samples but different seed.",
                    category=UserWarning,
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    "Deserializing a MetropolisSamplerState saved with a different number of MPI nodes and different number"
                    f"of chains. This is impossible. Use n_chains={n_chains_serialized} to load this file correctly."
                    "\nInstead of hard erroring, we will skip loading the sampler state.",
                    category=UserWarning,
                    stacklevel=2,
                )

            rank_mismatch = True
        else:
            rank_mismatch = False
            n_chains_serialized = n_chains

        del state_dict["distributed"]
        if not rank_mismatch:
            for prop in ["σ", "n_accepted_proc", "rng"]:
                x = state_dict[prop]
                y = x.reshape(mpi.n_nodes, *getattr(sampler_state, prop).shape)
                state_dict[prop] = y[mpi.rank]
        elif n_chains_serialized == n_chains:
            x = state_dict["σ"]
            y = x.reshape(mpi.n_nodes, n_chains_serialized, *x.shape[1:])
            state_dict["σ"] = y[mpi.rank]
            state_dict["n_accepted_proc"] = sampler_state.n_accepted_proc
            state_dict["rng"] = sampler_state.rng
        else:
            state_dict["σ"] = sampler_state.σ
            state_dict["n_accepted_proc"] = sampler_state.n_accepted_proc
            state_dict["rng"] = sampler_state.rng

    return MetropolisSamplerState._from_flax_state_dict(
        MetropolisSamplerState._pytree__static_fields, sampler_state, state_dict
    )


# # when running on multiple jax processes the σ and n_accepted_proc are not fully addressable
# # however, when serializing they need to be so here we register custom handlers which
# # gather all the data to every process.
# # when deserializing we distribute the samples again to all availale devices
# # this way it is enough to serialize on process 0, and we can restart the simulation
# # also  on a different number of devices, provided the number of samples is still
# # divisible by the new number of devices
# serialization.register_serialization_state(
#     MetropolisSamplerState,
#     serialize_MetropolisSamplerState,
#     deserialize_MetropolisSamplerState,
#     override=True,
# )
