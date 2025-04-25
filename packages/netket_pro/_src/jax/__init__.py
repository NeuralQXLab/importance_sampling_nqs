# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from netket_pro._src.jax._chunk_utils import chunk, unchunk
from netket_pro._src.jax._scanmap import (
    scan_reduce,
    scan_append,
    scan_append_reduce,
    scanmap,
)
from netket_pro._src.jax._vjp_chunked import vjp_chunked
from netket_pro._src.jax._vmap_chunked import apply_chunked, vmap_chunked
from netket_pro._src.jax.expect import expect_advanced

from netket_pro._src.jax.eigh import eigh
from netket_pro._src.jax.pfaffian import log_pfaffian

from netket_pro._src.jax.pallas_ops import vec_to_tril

from netket_pro._src.jax.lax import reduce_xor
from netket_pro._src.jax._sparse import COOTensor
