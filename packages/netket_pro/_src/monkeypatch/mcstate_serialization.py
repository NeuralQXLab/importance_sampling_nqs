from flax import serialization

from netket.vqs.mc.mc_state.state import (
    serialize_MCState as _nk_serialize_MCState,
    deserialize_MCState as _nk_deserialize_MCState,
)
from netket.vqs.mc.mc_state.state import MCState


def _is_default_only_sampler(vstate):
    distributions = list(vstate.sampler_states.keys())
    if len(distributions) == 1:
        if distributions[0] == "default":
            return True
    return False


# serialization
def serialize_MCState(vstate):
    """
    Overwrite the netket serialization for MCState to include the sampler states of extra distributions.
    """
    state_dict = _nk_serialize_MCState(vstate)

    if _is_default_only_sampler(vstate):
        return state_dict

    sampler_states = {}
    for key in vstate.sampler_states.keys():
        # Do not double serialize the default sampler
        if key == "default":
            continue

        samples = vstate._samples_distributions.get(key, None)
        if samples is not None:
            sampler_states[key] = vstate._sampler_states_previous[key]
        else:
            sampler_states[key] = vstate.sampler_states[key]

    state_dict["sampler_states"] = serialization.to_state_dict(sampler_states)
    return state_dict


def deserialize_MCState(vstate, state_dict):
    res = _nk_deserialize_MCState(vstate, state_dict)
    if "sampler_states" in state_dict:
        ss = res.sampler_state
        for key, value in state_dict["sampler_states"].items():
            res.sampler_states[key] = serialization.from_state_dict(ss, value)

    return res


serialization.register_serialization_state(
    MCState,
    serialize_MCState,
    deserialize_MCState,
    override=True,
)
