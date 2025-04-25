from functools import partial

import sparse

import jax
import jax.numpy as jnp

import numpy as np

from flax import struct

from netket.operator import DiscreteJaxOperator
from netket.hilbert import SpinOrbitalFermions
from netket.utils.types import PyTree

from netket_pro._src.operator.particle_number_conserving_fermionic import (
    _get_conn_padded,
    _jw_kernel,
    _prepare_operator_data_from_coords_data_dict,
    split_diag_offdiag,
    prepare_data,
    prepare_data_diagonal,
    _collect_ops,
    _fermiop_terms_to_sites_daggers_weights,
)
from netket_pro._src.operator.pyscf_utils import (
    compute_pyscf_integrals,
    to_desc_order_sparse,
)
from netket_pro._src.operator.normal_order_utils import (
    move_daggers_left,
    to_desc_order,
    to_normal_order,
)


# TODO do this in hilbert
@partial(jax.jit, static_argnames="n_spin_subsectors")
def unpack_du(x, n_spin_subsectors=2):
    assert x.shape[-1] % n_spin_subsectors == 0
    x_ = x.reshape(x.shape[:-1] + (n_spin_subsectors, x.shape[-1] // n_spin_subsectors))
    return tuple(x_[..., i, :] for i in range(n_spin_subsectors))


@jax.jit
def pack_du(*xs):
    xs = jnp.broadcast_arrays(*xs)
    xd = xs[0]
    n_spin_subsectors = len(xs)
    res = jnp.zeros(
        xd.shape[:-1]
        + (
            n_spin_subsectors,
            xd.shape[-1],
        ),
        dtype=xd.dtype,
    )
    for i, xi in enumerate(xs):
        res = res.at[..., i, :].set(xi)
    return jax.lax.collapse(res, res.ndim - 2, res.ndim)


@partial(jax.jit, static_argnums=(0, 1))
@partial(jnp.vectorize, signature="(n),(n)->(m,n),(m,n),(m)", excluded=(0, 1, 4, 5, 6))
def _get_conn_padded_interaction_up_down(
    nelectron_down, nelectron_up, x_down, x_up, index_array, create_array, weight_array
):
    dtype = x_down.dtype
    # n_orbitals = x_down.shape[-1]

    assert x_down.ndim == 1
    if index_array is not None:
        assert index_array.ndim == 2
    else:  # diagonal
        assert weight_array.ndim == 2

    (down_occupied,) = jnp.where(x_down, size=nelectron_down)
    (up_occupied,) = jnp.where(x_up, size=nelectron_up)

    k_destroy_down, k_destroy_up = jnp.meshgrid(down_occupied, up_occupied)

    if index_array is None:  # diagonal
        weight = weight_array[k_destroy_down, k_destroy_up]
        xp_down = x_down[None, :]
        xp_up = x_up[None, :]
        sign = 1
        mels = sign * weight.sum()[None]
    else:
        ind = index_array[k_destroy_down, k_destroy_up].ravel()
        weight = weight_array[ind]
        l_create = create_array[ind]

        k_destroy_down = k_destroy_down.reshape(1, -1)
        k_destroy_up = k_destroy_up.reshape(1, -1)
        l_create_down = l_create[..., :1]
        l_create_up = l_create[..., 1:]

        xp_down, sign_down, down_create_is_not_occupied = _jw_kernel(
            k_destroy_down, l_create_down, x_down
        )
        xp_up, sign_up, up_create_is_not_occupied = _jw_kernel(
            k_destroy_up, l_create_up, x_up
        )

        up_is_diagonal = k_destroy_up[0][:, None] == l_create_up[..., 0]
        down_is_diagonal = k_destroy_down[0][:, None] == l_create_down[..., 0]
        both_not_occupied = (down_create_is_not_occupied | down_is_diagonal) & (
            up_create_is_not_occupied | up_is_diagonal
        )

        sign = sign_up * sign_down
        mels = weight * both_not_occupied * sign

        xp_down = jnp.where((mels == 0)[:, :, None], x_down[None, None, :], xp_down)
        xp_up = jnp.where((mels == 0)[:, :, None], x_up[None, None, :], xp_up)

        xp_down = jax.lax.collapse(xp_down, 0, xp_down.ndim - 1).astype(dtype)
        xp_up = jax.lax.collapse(xp_up, 0, xp_up.ndim - 1).astype(dtype)
        mels = jax.lax.collapse(mels, 0, mels.ndim)
    return xp_down, xp_up, mels


@partial(jax.jit, static_argnames=("nelec",))
def get_conn_padded_pnc_spin(_operator_data, x, nelec):
    dtype = x.dtype
    if not jnp.issubdtype(dtype, jnp.integer) or jnp.issubdtype(dtype, jnp.integer):
        x = x.astype(jnp.int8)

    n_spin_subsectors = len(nelec)
    xs = unpack_du(x, n_spin_subsectors)
    xs_diag = tuple(a[..., None, :] for a in xs)

    xp_list = []
    mels_list = []
    xp_diag = None
    mels_diag = 0

    # TODO make sectors a jax array and use jax loop here to compile only once
    # (requires nelec to be the same for all sectors)

    for (k, sectors), v in _operator_data["diag"].items():
        if k == 0:
            assert sectors == ()
            sectors = (0,)  # dummy sector

        for i in sectors:
            _, melsi = _get_conn_padded(nelec[i], xs[i], *v)
            mels_diag = mels_diag + melsi
            xp_diag = x[..., None, :]

    for (k, sectors), v in _operator_data["mixed_diag"].items():
        if k != 4:
            raise NotImplementedError
        # TODO make sectors a jax array and use jax loop here to compile only once
        for i, j in sectors:
            assert i > j  # here i>j
            # e.g. take operator data to be c_ijkl + c_jilk so that here we only need to sum  ρ > σ (i.e. σ=d, ρ=u)
            *_, melsij = _get_conn_padded_interaction_up_down(
                nelec[j], nelec[i], xs[j], xs[i], *v
            )
            mels_diag = mels_diag + melsij
            xp_diag = x[..., None, :]

    # TODO optionally always add zero diagonal?
    if xp_diag is not None:
        xp_list = [xp_diag]
        mels_list = [mels_diag]

    for (k, sectors), v in _operator_data["offdiag"].items():
        # TODO make sectors a jax array and use jax loop here to compile only once
        for i in sectors:
            xpi, melsi = _get_conn_padded(nelec[i], xs[i], *v)
            xpi = pack_du(*xs_diag[:i], xpi, *xs_diag[i + 1 :])
            xp_list.append(xpi)
            mels_list.append(melsi)

    for (k, sectors), v in _operator_data["mixed_offdiag"].items():
        if k != 4:
            raise NotImplementedError
        for i, j in sectors:
            assert i > j  # here i>j
            # e.g. take operator data to be c_ijkl + c_jilk so that here we only need to sum  ρ > σ (i.e. σ=d, ρ=u)
            xpj, xpi, melsij = _get_conn_padded_interaction_up_down(
                nelec[j], nelec[i], xs[j], xs[i], *v
            )
            xpij = pack_du(
                *xs_diag[:j], xpj, *xs_diag[j + 1 : i], xpi, *xs_diag[i + 1 :]
            )
            xp_list.append(xpij)
            mels_list.append(melsij)
    if len(xp_list) > 0:
        xp = jnp.concatenate(xp_list, axis=-2).astype(dtype)
        mels = jnp.concatenate(mels_list, axis=-1)
    else:
        xp = jnp.zeros((*x.shape[:-1], 0, x.shape[-1]), dtype=dtype)
        mels = jnp.zeros(xp.shape[:-1])  # TODO dtype?
    return xp, mels


def _split_spin_sectors(sites, daggers, weights, n_orbitals, n_spin_subsectors):
    n_ops = sites.shape[1]
    if n_ops == 0:
        return sites, np.zeros_like(sites), daggers, weights
    L = np.arange(n_spin_subsectors) * n_orbitals
    R = np.arange(1, n_spin_subsectors + 1) * n_orbitals
    sectors_mask = (sites[..., None] >= L) & (
        sites[..., None] < R
    )  # n_terms x n_ops x n_spin_subsectors
    sectors = np.einsum("...i,i", sectors_mask, np.arange(n_spin_subsectors)).astype(
        np.int32
    )
    sites = sites - sectors * n_orbitals
    return sites, sectors, daggers, weights


def split_spin_sectors(d, n_orbitals, n_spin_subsectors):
    # input: { size : (sites, daggers, weights) }
    # output: { size : (sites, sectors, daggers, weights) }
    return {
        k: _split_spin_sectors(*v, n_orbitals, n_spin_subsectors) for k, v in d.items()
    }


def _merge_spin_sectors(sites, sectors, daggers, weights, n_orbitals):
    return sites + sectors * n_orbitals, daggers, weights


def merge_spin_sectors(d, n_orbitals):
    # input: { size : (sites, sectors, daggers, weights) }
    # output: { size : (sites, daggers, weights) }
    return {k: _merge_spin_sectors(*v, n_orbitals) for k, v in d.items()}


def _fermiop_terms_to_sites_sectors_daggers_weights(
    terms, weights, n_orbitals, n_spin_subsectors
):
    # output: { size : (sites, sectors, daggers, weights) }
    return split_spin_sectors(
        _fermiop_terms_to_sites_daggers_weights(terms, weights),
        n_orbitals,
        n_spin_subsectors,
    )


def sites_daggers_weights_to_sparse(sites, daggers, weights, n_orbitals):
    n = daggers.shape[-1]
    assert n % 2 == 0
    assert (daggers[:, : n // 2] == 1).all()
    assert (daggers[:, n // 2 :] == 0).all()
    # TODO cutoff?
    return sparse.COO(sites.T, weights, shape=(n_orbitals,) * n)


def extract_operators_normal_order(*sites_daggers_weights, n_orbitals):
    operators = []
    while True:
        sites_daggers_weights_left, sites_daggers_weights = move_daggers_left(
            *sites_daggers_weights
        )
        sites_daggers_weights_left = to_desc_order(*sites_daggers_weights_left)
        o = sites_daggers_weights_to_sparse(*sites_daggers_weights_left, n_orbitals)
        operators.append(o)
        if sites_daggers_weights[0] is None:
            break
        elif sites_daggers_weights[0].shape[-1] == 0:
            operators.append(sites_daggers_weights[2].sum())
            break
    return operators


def arrays_to_fermiop_terms(t):
    terms = []
    weights = []
    for s, d, w in t.values():
        terms = terms + np.concatenate([s[..., None], d[..., None]], axis=-1).tolist()
        weights = weights + w.tolist()
    return terms, weights


# version for the one with sectors for the si, taking a dict
def _sparse_arrays_to_coords_data_dict2(ops):
    def _unpack(k, v):
        if isinstance(k, tuple):
            k, _ = k
        if k == 0:
            return np.zeros((1, 0), dtype=int), np.array([v])
        else:
            return v.coords.T, v.data

    return {k: _unpack(k, v) for k, v in ops.items()}


def prepare_operator_data_from_coords_data_dict_spin(coords_data_sectors, n_orbitals):
    # version with sectors
    _cond = lambda s: s == () or _len(s[0]) == 1
    coords_data_same = {
        (k, s): v for (k, s), v in coords_data_sectors.items() if _cond(s)
    }
    coords_data_mixed = {
        (k, s): v for (k, s), v in coords_data_sectors.items() if not _cond(s)
    }

    operator_data = _prepare_operator_data_from_coords_data_dict(
        coords_data_same, n_orbitals
    )
    # process mixed terms
    data_diag_mixed = {}
    data_offdiag_mixed = {}
    for k, v in coords_data_mixed.items():
        sw_diag, sw_offdiag = split_diag_offdiag(*v)
        if len(sw_diag[-1]) > 0:
            data_diag_mixed[k] = prepare_data_diagonal(
                *sw_diag, n_orbitals, _sparse=False
            )
        if len(sw_offdiag[-1]) > 0:
            data_offdiag_mixed[k] = prepare_data(*sw_offdiag, n_orbitals, _sparse=False)
    operator_data = {
        **operator_data,
        "mixed_diag": data_diag_mixed,
        "mixed_offdiag": data_offdiag_mixed,
    }
    return operator_data


# TODO generalize it to >4 fermionic operators


def _len(s):
    if isinstance(s, tuple):
        return len(s)
    else:
        return 1


class ParticleNumberConservingFermioperator2ndSpinJax(DiscreteJaxOperator):
    """
    Efficient implemnetation of the fermionic operator in the case of particle number conservation.
    Jax version.

    .. warning::

        This operator should not be constructed directly! Use the appropriate factory methods.

    .. note::

        This operator was implemented by the god of Jax himself, Clemens.
        Do offer him a beer.

    This operator internally implements something like

    .. math::

        H = a + Σ_ijσ b_ij c_iσ^† c_jσ + Σ_ijklσρ c_ijkl  c_iσ^† c_jρ^† c_kρ c_lσ

    """
    def __init__(self, 
                 hilbert:SpinOrbitalFermions,
                 operator_data, 
                 max_conn_size):

        self._hilbert = hilbert
        self._operator_data = operator_data
        self._max_conn_size = max_conn_size

    def tree_flatten(self):
        # Children are the dynamic values to be treated as leaves in the pytree.
        children = (self._operator_data,)
        # Static values are the auxiliary data that help in reconstructing the object.
        aux_data = (self._hilbert, self._max_conn_size)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        _hilbert, _max_conn_size = aux_data
        _operator_data, = children
        return cls(_hilbert, _operator_data, _max_conn_size)

    @property
    def dtype(self):
        return NotImplemented

    @property
    def is_hermitian(self):
        # TODO actually check that it is
        return True

    @property
    def max_conn_size(self):
        return self._max_conn_size

    def get_conn_padded(self, x):
        return get_conn_padded_pnc_spin(
            self._operator_data, x, self._hilbert.n_fermions_per_spin
        )

    @classmethod
    def from_coords_data(cls, hilbert, coords_data_sectors):
        assert isinstance(hilbert, SpinOrbitalFermions)
        assert hilbert.n_fermions is not None
        assert hilbert.n_spin_subsectors >= 2
        n_orbitals = hilbert.n_orbitals
        operator_data = prepare_operator_data_from_coords_data_dict_spin(
            coords_data_sectors, n_orbitals
        )

        # Compute max_conn_size
        x = jax.ShapeDtypeStruct((1, hilbert.size), dtype=jnp.uint8)
        _fun = partial(get_conn_padded_pnc_spin, nelec=hilbert.n_fermions_per_spin)
        _, mels = jax.eval_shape(_fun, operator_data, x)
        max_conn_size = mels.shape[-1]

        return cls(hilbert, operator_data, max_conn_size)

    @classmethod
    def from_sparse_arrays(cls, hilbert, operators_sector):
        # operators_sector is a dict with
        # key: a tuple (k, sectors)
        #      where k is the number of c/c^†
        #      and sectors is a tuple of tuples/numbers containing the index / sets of indices of sectors the operator is acting on
        #      each element of sectors needs to be ordered in descending order
        # value: a sparse matrix of coefficeints of shape (n_orbitals,)*k

        # convert sparse arrays to coords+data tuple
        coords_data_sectors = _sparse_arrays_to_coords_data_dict2(operators_sector)
        return cls.from_coords_data(hilbert, coords_data_sectors)

    @classmethod
    def from_sparse_arrays_all_sectors(cls, hilbert, operators, cutoff=1e-11):
        # operators = [const, hij, hijkl]
        # ops = {0: const, 2: hij_sparse, 4: hijkl_sparse}
        ops = _collect_ops(operators)

        operators_sector = {}
        sectors0 = ()
        sectors1 = tuple(np.arange(hilbert.n_spin_subsectors).tolist())
        sectors2 = tuple(
            map(
                tuple,
                np.array(np.tril_indices(hilbert.n_spin_subsectors, -1)).T.tolist(),
            )
        )
        for k, v in ops.items():
            if k == 0:
                operators_sector[0, sectors0] = v
            elif k == 2:
                operators_sector[2, sectors1] = v
            elif k == 4:
                operators_sector[4, sectors1] = to_desc_order_sparse(v, cutoff)
                # add c_ijkl + c_jilk
                # Σ_{σ!=ρ} c_ijkl  c_iσ^† c_jρ^† c_kρ c_lσ =  Σ_{σ>ρ} (c_ijkl + c_jilk) c_iσ^† c_jρ^† c_kρ c_lσ
                operators_sector[4, sectors2] = v.swapaxes(2, 3) + v.swapaxes(0, 1)
            else:
                raise NotImplementedError
        return cls.from_sparse_arrays(hilbert, operators_sector)

    @classmethod
    def from_pyscf_molecule(cls, mol, mo_coeff, cutoff=1e-11):
        n_orbitals = int(mol.nao)
        hilbert = SpinOrbitalFermions(
            n_orbitals, s=1 / 2, n_fermions_per_spin=mol.nelec
        )

        const, hij, hijkl = compute_pyscf_integrals(
            mol, mo_coeff
        )  # not in normal order
        hij = hij * (jnp.abs(hij) > cutoff)
        hij_sparse = sparse.COO.from_numpy(hij)
        hijkl = hijkl * (jnp.abs(hijkl) > cutoff)
        hijkl_sparse = 0.5 * sparse.COO.from_numpy(hijkl)
        return cls.from_sparse_arrays_all_sectors(
            hilbert, [const, hij_sparse, hijkl_sparse], cutoff=cutoff
        )

    @classmethod
    def from_sites_sectors_daggers_weights(cls, hilbert, t, cutoff=1e-11):
        # t: { size : (sites, sectors, daggers, weights) }
        # arbitrary order of sites, sectors, and daggers
        # is internally converted to the right order for the operator
        n_orbitals = hilbert.n_orbitals
        n_spin_subsectors = hilbert.n_spin_subsectors
        tno = _to_tno_sector(t, n_spin_subsectors, n_orbitals)
        operators_sector = _tno_sector_to_operators_sector(
            tno, n_spin_subsectors, n_orbitals, cutoff=cutoff
        )
        return cls.from_sparse_arrays(hilbert, operators_sector)

    @classmethod
    def from_fermiop(cls, ha, cutoff=1e-11):
        """
        Constructs a ParticleNumberConservingFermioperator2ndSpinJax operator from a
        :class:`~netket.operator.FermionOperator2ndJax`.

        Args:
            ha: A :class:`~netket.operator.FermionOperator2ndJax` operator.
            kwargs: Additional keyword arguments to pass to the constructor (internal use only).
        """

        hilbert = ha.hilbert
        n_orbitals = hilbert.n_orbitals
        n_spin_subsectors = hilbert.n_spin_subsectors
        t = _fermiop_terms_to_sites_sectors_daggers_weights(
            ha.terms, ha.weights, n_orbitals, n_spin_subsectors
        )
        return cls.from_sites_sectors_daggers_weights(hilbert, t, cutoff=cutoff)


def _to_tno_sector(t, n_spin_subsectors, n_orbitals):
    # convert to normal order with higher sector to the left
    return split_spin_sectors(
        to_normal_order(merge_spin_sectors(t, n_orbitals)),
        n_orbitals,
        n_spin_subsectors,
    )


def _tno_sector_to_operators_sector(
    tno_sector, n_spin_subsectors, n_orbitals, cutoff=1e-11
):
    def _insert_append(d, k, s, o, cutoff):
        # check if an element with the same matrix but different sectors exist
        # if yes append to the list of sectors
        # else insert new element into the dict
        for (k2, s2), o2 in d.items():
            # and same number of sectors, same number of fermionic operators, same matrix (up to cutoff)
            if (
                (
                    (s == () and s2 == ())
                    or (len(s2) > 0 and len(s) > 0 and _len(s2[0]) == _len(s[0]))
                )
                and k == k2
                and sparse.abs(o - o2).max() < cutoff
            ):
                d[k, s2 + s] = d.pop((k2, s2))
                break
        else:
            d[k, s] = o

    operators_sector = {}

    for k, (sites, sectors, daggers, weights) in tno_sector.items():
        for i in range(n_spin_subsectors):
            if not (((2 * daggers - 1) * (sectors == i)).sum(axis=-1) == 0).all():
                raise ValueError  # does not conserve particle number per sector

        sector_count = jax.vmap(partial(jnp.bincount, length=n_spin_subsectors))(
            sectors
        )

        # merge sectors which have same sparse matrix

        if k == 0:
            operators_sector[0, ()] = weights.reshape(())
        elif k == 2:
            # at this point we know there is only one sector this acts on
            sector = sectors[:, 0]  # = sectors[:, 1]
            for i in np.unique(sector):
                m = sector == i
                o = sites_daggers_weights_to_sparse(
                    sites[m], daggers[m], weights[m], n_orbitals=n_orbitals
                )
                _insert_append(operators_sector, k, (i,), o, cutoff)
        elif k == 4:
            # at this point we know that n_sectors_acting_on \in 1,2
            n_sectors_acting_on = np.count_nonzero(sector_count, axis=-1)

            # all same sector
            m_same = n_sectors_acting_on == 1
            sector = sectors[:, 0]
            for i in np.unique(sector[m_same]):
                m = (sector == i) & m_same
                o = sites_daggers_weights_to_sparse(
                    sites[m], daggers[m], weights[m], n_orbitals=n_orbitals
                )
                _insert_append(operators_sector, k, (i,), o, cutoff)

            m_different = ~m_same
            sector = sectors[:, :2]
            # i > j because we made it normal order (with site shifted by N*spin) above
            for ij in np.unique(sector[m_different], axis=0):
                m = (sector == ij[None]).all(axis=-1) & m_different
                # minus sign because in the operator (_get_conn_padded_interaction_up_down) we assume it's swaped to (assuming σ>ρ)
                # cσ^† cσ cρ^† cρ = - cσ^† cρ^† cσ cρ
                o = -sites_daggers_weights_to_sparse(
                    sites[m], daggers[m], weights[m], n_orbitals=n_orbitals
                )
                _insert_append(operators_sector, k, (tuple(ij),), o, cutoff)
        else:
            raise NotImplementedError
    return operators_sector

jax.tree_util.register_pytree_node(
    ParticleNumberConservingFermioperator2ndSpinJax,
    ParticleNumberConservingFermioperator2ndSpinJax.tree_flatten,
    ParticleNumberConservingFermioperator2ndSpinJax.tree_unflatten
)