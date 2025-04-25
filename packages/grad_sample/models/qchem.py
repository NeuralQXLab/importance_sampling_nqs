import netket as nk
import pubchempy as pcp
from pyscf import gto, scf, fci, cc, mcscf, ci
import netket.experimental as nkx
from netket_pro.operator import ParticleNumberConservingFermioperator2ndSpinJax
import jax.numpy as jnp
import networkx as nx

class PCMolecule:
    """
    Initialize a model from a PubChem CID
    Defaults
    """
    def __init__(self, cid=947, use_nat_orb=False):
        # 1. Load the N2 molecule from PubChem (3D structure)
        self.cid = cid  # N2 molecule
        self.use_nat_orb = use_nat_orb
        try:
            self.compound = pcp.get_compounds(cid, "cid", record_type='3d')[0]
            self.geom = '3d'
        except : 
            print('using 2d')
            self.compound = pcp.get_compounds(cid, "cid", record_type='2d')[0]
            self.geom = '2d'
        self.name = cid
        # 2. Extract atomic coordinates
        geometry = []
        
        for atom in self.compound.atoms:
            symbol = atom.element
            if self.geom == '3d':
                x, y, z  = atom.x, atom.y, atom.z
            elif self.geom == '2d':
                x, y, z  = atom.x, atom.y, 0.0
            geometry.append(f"{symbol} {x} {y} {z}")

        # Convert to PySCF format
        mol_geometry = "\n".join(geometry)

        # 3. Define the molecule in PySCF
        mol = gto.Mole()
        mol.atom = mol_geometry
        mol.basis = "STO-3G"  # Choose a reasonable basis set
        mol.unit = "angstrom"  # Coordinates are in Ångströms
        mol.spin = 0  # N2 is a singlet
        mol.charge = 0
        mol.build()

        # 4. Run Hartree-Fock calculation
        mf = scf.HF(mol)
        mf.kernel()

        cisd = ci.CISD(mf)
        cisd.kernel()
        cisd_energy = cisd.e_tot
        cisd_amplitudes = cisd.ci

        # 5. Compute Full Configuration Interaction (FCI) energy 
        ccsd = cc.ccsd.CCSD(mf).run()
        nat_orbs = mcscf.addons.make_natural_orbitals(ccsd)

        # natorbital hamiltonian
        if self.use_nat_orb:
            ha_pyscf = nkx.operator.from_pyscf_molecule(mol, mo_coeff=nat_orbs[1]).to_jax_operator()
        else:
            ha_pyscf = nkx.operator.from_pyscf_molecule(mol).to_jax_operator()

        self.hamiltonian = ParticleNumberConservingFermioperator2ndSpinJax.from_fermiop(ha_pyscf)
        self.hilbert_space = self.hamiltonian.hilbert
        
        mo_coeff = mf.mo_coeff

        # make sampling graph
        E_nuc, Tij, Vijkl = nkx.operator.pyscf.TV_from_pyscf_molecule(mol, mo_coeff=mo_coeff)
        No = Tij.shape[0]//2
        Tij = Tij.todense()
        Tij_up = Tij[:No, :No]
        Tij_down = Tij[No:, No:]
        T_tot = jnp.abs(Tij_up) + jnp.abs(Tij_down)
        T_tot_od = jnp.abs(T_tot-jnp.diag(jnp.diagonal(T_tot)))

        Vijkl = Vijkl.todense()
        Vt = jnp.zeros((2*No, 2*No))
        for i in range(2*No):
            Vt = Vt + jnp.abs(Vijkl[i,:,i,:])
        Vt_od = jnp.abs(Vt-jnp.diag(jnp.diagonal(Vt)))
        Vt2 = Vt_od[:No, :No]+Vt_od[No:, No:]

        # Compute the combined adjacency matrix based on the condition
        combined_adjacency = (Vt2 + T_tot) > 0
        combined_adjacency = combined_adjacency.at[jnp.diag_indices_from(combined_adjacency)].set(0)
        adj_matrix = jnp.array(combined_adjacency).astype(int)

        # Create a NetworkX graph from the adjacency matrix
        G = nx.from_numpy_array(adj_matrix)

        # connect medians of each cluster to each other
        components = list(nx.connected_components(G))
        medians = []
        for comp in components:
            sorted_nodes = sorted(comp)
            median_idx = len(sorted_nodes) // 2
            median_node = sorted_nodes[median_idx]
            medians.append(median_node)
        medians.sort()
        for i in range(len(medians) - 1):
            G.add_edge(medians[i], medians[i + 1])
            
        g = nk.graph.Graph.from_networkx(G)
        # g = nk.graph.Chain(self.hilbert_space.n_orbitals, pbc=False)
        self.graph = nk.graph.disjoint_union(g, g) #only relevant for the fermihop sampler
        
        # define dummy vars for automatic naming
        self.Ns = self.hilbert_space.size
        self.h = 0
        self.L = self.Ns
    
        # 5. Compute Full Configuration Interaction (FCI) energy
        # self.fci_dict = {166630 : -87.892693, 947: -107.660206}
        self.fci_dict = {166630 : -87.89289424618677, 947: -107.660206}
        if self.cid in self.fci_dict.keys():
            self.E_fci = self.fci_dict[self.cid]
        else:
            cisolver = fci.FCI(mol, mf.mo_coeff)
            self.E_fci = cisolver.kernel()[0]