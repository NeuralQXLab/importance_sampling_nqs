import netket as nk

class TFI:
    def __init__(self, L=3, h=1.0):
        self.Ns = L * L
        self.h = h
        self.L = L
        self.name = "ising"
        # lattice = nk.graph.Square(L, max_neighbor_order=2)

        self.graph = nk.graph.Square(L, pbc=True)

        self.hilbert_space = nk.hilbert.Spin(s=1 / 2, N=self.graph.n_nodes, inverted_ordering=False)
        
        self.hamiltonian = nk.operator.Ising(hilbert=self.hilbert_space, graph=self.graph, h=self.h)
