from functools import lru_cache
from typing import Optional

import numpy as np

import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils

from netket import config as nkconfig
from netket import jax as nkjax
from netket.utils import mpi
from jax.lax import with_sharding_constraint
from jax.sharding import PositionalSharding


@lru_cache
def mode() -> str:
    """
    Returns the distributed mode used by NetKet.

    This can be one of the following: ``None``, ``"sharding"``, or ``"mpi"``.
    """
    if nkconfig.netket_experimental_sharding:
        return "sharding"
    elif process_count() > 1:
        return "mpi"
    else:
        return None


@lru_cache
def process_count() -> int:
    """
    Returns the total number of JAX processes running NetKet.

    If you are running with experimental sharding, this is
    equivalent to ``jax.process_count()``. If you are running
    with mpi, this is ``nk.utils.mpi.n_nodes``.
    """

    if nkconfig.netket_experimental_sharding:
        return jax.process_count()
    else:
        return mpi.n_nodes


@lru_cache
def device_count() -> int:
    """
    Returns total number of devices.
    """
    if mode() == "sharding":
        return jax.device_count()
    else:
        return process_count()


@lru_cache
def process_index() -> int:
    """
    Returns the index of this process running NetKet.

    If you are running with experimental sharding, this is
    equivalent to ``jax.process_index()``. If you are running
    with mpi, this is ``nk.utils.mpi.rank``.

    This is an integer between 0 and
    :func:`netket_pro.distributed.process_count()`.
    """

    if nkconfig.netket_experimental_sharding:
        return jax.process_index()
    else:
        return mpi.rank


def is_master_process() -> bool:
    """
    Returns whether the current process is the master process.
    """
    return process_index() == 0


def broadcast_key(key: Optional[jax.Array] = None, *, root: int = 0) -> jax.Array:
    """
    Given a `jax.random.key`, distribute it among all nodes.
    """
    return nkjax.PRNGKey(key, root=root)


def broadcast(array: jax.Array, *, root: int):
    """
    Broadcasts an array from the root process to all other processes, giving a replicated
    array.

    The input array on non-root processes must be a dummy array with
    the same shape as the array on the root process.

    Args:
        array: The array to broadcast. On non-root processes, this should be a
            placeholder array with the right shape and dtype.
        root: The root process that holds the original array.
    """
    _mode = mode()
    if _mode == "sharding":
        result = multihost_utils.broadcast_one_to_all(
            array, is_source=jax.process_index() == root
        )
    elif _mode == "mpi":
        result, _ = mpi.mpi_bcast_jax(array, root=root)
    else:
        result = array
    return result


def shard_replicated(array, *, axis=0):
    """
    Shards a replicated array across MPI ranks/jax processes.

    The input must be a replicated array, obtained either from
    :func:`netket_pro.distributed.broadcast`, :func:`netket_pro.distributed.allgather` or
    from executing the same function on all nodes.

    When running under MPI, the output is simply a slice of the input array along the
    specified axis (Default 0) corresponding to the rank of the process.

    When running under sharding, we set the sharding constraint accordingly.

    Args:
        array: The array to shard. Must be replicated!
        axis: The axis along which to shard (Default 0).
    """

    def _shard(array):
        lenght = array.shape[axis]
        if not lenght % process_count() == 0:
            raise ValueError(
                "Sharded axis size must be a multiple of the number of processes"
            )

        if mode() == "sharding":
            # Do not use process_count() because we could have more than
            # 1 GPU per process

            sharding_shape = [1 for _ in range(array.ndim)]
            sharding_shape[axis] = len(jax.devices())
            sharding = jax.sharding.PositionalSharding(jax.devices()).reshape(
                sharding_shape
            )
            array = jax.lax.with_sharding_constraint(array, sharding)
        elif mode() == "mpi":
            lenght_per_proc = lenght // mpi.n_nodes
            start, end = mpi.rank * lenght_per_proc, (mpi.rank + 1) * lenght_per_proc
            array = array[start:end]
        else:
            pass
        return array

    return jax.tree_util.tree_map(_shard, array)


def declare_replicated_array(x):
    """
    Declares that an array is replicated across all processes.

    This should be used when we build an array by 'hand' and we know it's the same on
    every process, but by default jax does not know that it is the same everywhere.
    So we declare explicitly that it is replcated.

    .. note::

        This only does something when sharding with 1 device per process. Does nothing
        otherwise.

    Args:
        x: The array to declare as replicated.

    Returns:
        An array with the same shape as the input, but declared as replicated.
    """
    if mode() == "sharding" and process_count() == device_count():
        par_sharding = jax.sharding.PositionalSharding(jax.devices()).replicate()

        return jax.make_array_from_single_device_arrays(x.shape, par_sharding, [x])
    else:
        return x


def allgather(array, *, axis: int = 0, token=None):
    """
    Gathers (unshard) a distributed (sharded) array to all processes.

    The resulting array will have the same shape as the input array except
    the first axis, which will be :ref:`netket_pro.distributed.process_count`
    times longer.

    .. note::

        An input array of shape :math:`(M, N, ...)` will lead to a gathered
        array of shape :math:`(P \times M, N, ...)`, where :math:`P` is the
        number of processes.

    .. note::

        The resulting array will be unsharded, or fully addressable locally
        and on every process.

    Args:
        array: The array to gather.
        token: A token to be used for MPI communication.

    Returns:
        A tuple of the gathered array and the token.

    """
    if axis != 0:
        raise NotImplementedError("Only axis=0 is supported for now. Open a PR.")

    if mode() == "mpi":
        _a = array
        array, token = mpi.mpi_allgather_jax(array, token=token)
        array = jax.lax.collapse(array, 0, 2)
    elif mode() == "sharding":
        sharding = PositionalSharding(jax.devices()).replicate()
        sharding = sharding.reshape(tuple(1 for _ in range(array.ndim)))
        array = jax.lax.with_sharding_constraint(array, sharding)
    else:
        pass
    return array, token


def pad_axis_for_sharding(
    array: jax.Array, *, axis: int = 0, padding_value: float | jax.Array = 0
) -> jax.Array:
    """
    Pads an array along an axis to make it divisible by the number of processes.

    Args:
        array: The array to pad.
        axis: The axis along which to pad.
        padding_value: The value to use for padding.

    Returns:
        The padded array.
    """
    axis_size = array.shape[axis]
    n_devices = device_count()

    if axis_size % n_devices != 0:
        padded_axis_size = int(n_devices * np.ceil(axis_size / n_devices))
        padding_shape = [(0, 0) for _ in range(array.ndim)]
        padding_shape[axis] = (0, padded_axis_size - axis_size)

        array = jnp.pad(
            array,
            padding_shape,
            constant_values=padding_value,
        )
    return array


def reshard(
    array: jax.Array,
    *,
    sharded_axis: int = 0,
    out_sharded_axis: int = 1,
    token=None,
    pad: bool = False,
    pad_value: jax.Array = 0.0,
) -> jax.Array:
    """
    Reshards an array to distribute another axis among the processes. Equivalent to
    :ref:`~mpi4jax.mpi_alltoall` in MPI jargon.

    The input array is assumed to be sharded along axis `sharded_axis`, and the resulting
    array will be sharded along axis `out_sharded_axis`. The sharded axis will be collected
    while the output sharded axis will be distributed.

    .. note::

        If the input array has shape :math:`(x, y, z)` and the input sharded axis is `y`,
        and the output sharded axis is `x`, the resulting array will have shape :math:`(x, y*P, z/P)`.

    Args:
        array: The array to reshard / alltoall.
        sharded_axis: The axis to be collected.
        out_sharded_axis: The axis to be distributed.
        token: A token to be used for MPI communication.
        pad: Whether to pad the axis to be sharded to be a multiple of the number of processes. If this is
            set to `False`, the size of the sharded axis must be a multiple of the number of processes.
            (Default: `False`)
        pad_value: The value to use for padding. (Default: `0.0`)

    """
    assert sharded_axis != out_sharded_axis
    assert 0 <= sharded_axis < array.ndim
    assert 0 <= out_sharded_axis < array.ndim

    # Pad the number of parameters to be a multiple of the number of MPI nodes
    # -> (#n_nodes, np_padded)
    if array.shape[out_sharded_axis] % device_count() != 0:
        if pad:
            array = pad_axis_for_sharding(
                array, axis=out_sharded_axis, padding_value=pad_value
            )
        else:
            raise ValueError(
                "Sharded axis size must be a multiple of the number of processes"
            )

    if mode() == "mpi":
        # Create a new shape with the axis to be sharded split into two
        # (..., M, ...) -> (..., n, M/n, ...)
        new_shape = list(array.shape)
        new_shape.insert(out_sharded_axis, process_count())
        new_shape[out_sharded_axis + 1] = -1
        array = jnp.reshape(array, new_shape)

        # Move this axis to the position 0
        # (..., n, M/n, ...) -> (n, ..., M/n, ...)
        array = jnp.moveaxis(array, out_sharded_axis, 0)
        array, token = mpi.mpi_alltoall_jax(array, token=token)

        # After the alltoall, the sharded axis is not split between the
        # position 0 and the actual sharded axis, so we need to collapse them.
        # First we move the sharded axis back to its original position
        if sharded_axis != 0:
            array = jnp.moveaxis(array, 0, sharded_axis)

        # Then we collapse them
        array = jax.lax.collapse(array, sharded_axis, sharded_axis + 2)
    elif mode() == "sharding":
        del sharded_axis  # unused

        sharding = PositionalSharding(jax.devices())
        sharding_shape = list(1 for _ in range(array.ndim))
        sharding_shape[out_sharded_axis] = -1
        sharding = sharding.reshape(sharding_shape)
        array = with_sharding_constraint(array, sharding)
    return array, token


def barrier(name: str):
    """
    Synchronizes all processes. This function ensures that all processes reach this point
    before continuing. It uses either MPI or Sharding for synchronization.

    Args:
        name: A unique string to identify the synchronization point.
    """
    if mode() == "mpi":
        mpi.MPI_py_comm.barrier()
    elif mode() == "sharding":
        multihost_utils.sync_global_devices(name)


def broadcast_string(s: str, root: int = 0) -> str:
    def _encode_string_to_uint64_array(s):
        """Encodes a string into a NumPy array of uint64."""
        byte_data = s.encode("utf-8")  # Convert to bytes
        padding_size = (
            8 - len(byte_data) % 8
        ) % 8  # Compute padding to make it multiple of 8
        byte_data += b"\x00" * padding_size  # Pad with null bytes
        uint64_array = np.frombuffer(byte_data, dtype=np.uint64)  # Interpret as uint64
        return uint64_array, padding_size

    def _decode_uint64_array_to_string(uint64_array, padding_size):
        """Decodes a NumPy uint64 array back to a string."""
        byte_data = uint64_array.tobytes()  # Convert back to bytes
        return (
            byte_data[:-padding_size].decode("utf-8")
            if padding_size
            else byte_data.decode("utf-8")
        )

    if mode() == "sharding":
        if root != 0:
            raise ValueError("Only root=0 is supported in sharding mode")

        encoded_array, pad_size = _encode_string_to_uint64_array(s)
        encoded_array = multihost_utils.broadcast_one_to_all(encoded_array)
        pad_size = multihost_utils.broadcast_one_to_all(pad_size)
        s = _decode_uint64_array_to_string(encoded_array, pad_size)
    elif mode() == "mpi":
        s = mpi.MPI_py_comm.bcast(s, root=root)

    return s


def _inspect(name: str, x: jax.Array):
    """
    Internal function to inspect the sharding of an array. To be used for debugging inside
    of :ref:`jax.jit`-ted functions.

    Args:
        name: A string to identify the array, usually the name, but can contain anything else.
        x: The array
    """
    if mode() == "sharding":

        def _cb(y):
            if process_index() == 0:
                print(
                    f"{name}: shape={x.shape}, sharding_shape: {y.shape}, sharding:",
                    y,
                    flush=True,
                )

        jax.debug.inspect_array_sharding(x, callback=_cb)
