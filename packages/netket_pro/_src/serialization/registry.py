from functools import partial

# flake8: noqa: E402
from nqxpack._src.lib_v1.custom_types import (
    register_serialization,
)


# Graph

from nqxpack._src.registry.netket import serialize_fullsumstate, deserialize_vstate
