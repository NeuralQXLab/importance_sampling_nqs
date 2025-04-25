import jax
import jax.numpy as jnp
from jax.tree_util import tree_map, tree_flatten

def shape_tree(pytree):
    return tree_map(lambda g: g.shape, pytree)

def snr_tree(pytree, pytree_fs):
    #centered on the full summation value
    return tree_map(lambda g, g_fs: jnp.sqrt(g.shape[0]*jnp.abs(g_fs)**2/(jnp.var(g, axis=0))), pytree, pytree_fs)

def dagger_pytree(jac_pytree):
    return tree_map(lambda x: x.conj().T, jac_pytree)

def vjp_pytree(jac_pytree, vector):
    return tree_map(lambda jac_block: jnp.einsum("...i,i->...", jac_block, vector), jac_pytree)

def mul_pytree(jac_pytree, vector):
    return tree_map(lambda jac_block: jac_block*vector, jac_pytree)

def pytree_mean(pytree):
    """
    Compute the mean of all elements across all leaves of a pytree.
    
    Args:
        pytree: A JAX pytree (e.g., dictionary, list, tuple, etc.).
        
    Returns:
        float: The mean of all elements in the pytree.
    """
    # Flatten the pytree to get all leaves
    leaves, _ = jax.tree_util.tree_flatten(pytree)
    
    # Compute total sum and count of elements across all leaves
    total_sum = sum(jnp.sum(leaf) for leaf in leaves)
    total_count = sum(leaf.size for leaf in leaves)
    
    # Return the mean
    return total_sum / total_count
        
def flatten_tree_to_array(tree):
    """
    Flattens a pytree where each leaf is an array with the first dimension `N`.
    Produces a single array of shape (N, N_elem), where N is the batch dimension
    and N_elem is the sum of flattened dimensions of all leaves (excluding N).

    Parameters:
    - tree: The input pytree (e.g., nested dicts, lists, or tuples of arrays).

    Returns:
    - A jax.numpy.ndarray of shape (N, N_elem).
    """
    # Flatten the tree and extract leaves
    leaves, _ = tree_flatten(tree)
    
    # Ensure all leaves share the same batch dimension N
    batch_sizes = [leaf.shape[0] for leaf in leaves]
    if len(set(batch_sizes)) > 1:
        raise ValueError("All leaves must have the same first batch dimension (N).")
    
    N = batch_sizes[0]  # Shared batch dimension
    
    # Flatten each leaf (excluding the batch dimension) and concatenate
    flattened_leaves = [leaf.reshape(N, -1) for leaf in leaves]
    return jnp.concatenate(flattened_leaves, axis=1)