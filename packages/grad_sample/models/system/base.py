import netket as nk
from collections.abc import Sequence
import json


class Spin_Half:
    name = "None"
    graph_name = "None"
    hamiltonian_name = "None"
    local_spin = 1 / 2
    pbc = True

    def __init__(self, N: int, L: int, J: Sequence[float], sz_sector=0):
        # From netket 3.14, inverted ordering must be specified to keep the old
        # behaviour of +1 == spin down and -1 == spin up. In the future, the
        # default will be flipped.
        self.hilbert_space = nk.hilbert.Spin(
            s=self.local_spin, N=N, total_sz=sz_sector, inverted_ordering=True
        )
        self.J = J
        self.N = N
        self.L = L
        self.sz_sector = sz_sector

    def name_and_arguments_to_dict(self):
        """
        Convert the arguments for __init__ to a dict.
        Saving and then loading in this dict allows one to obtain the correct system from file.
        """
        return {
            "name": self.name,
            "L": self.L,
            "J": self.J,
            "sz_sector": self.sz_sector,
        }

    @staticmethod
    def argument_saver(
        arg_dict: dict, file_name: str, prefix: str = None, write_mode: str = "a"
    ):
        """
        Save the arg_dict to file fname
        """
        if not prefix == None:
            save_dict = {prefix: arg_dict}
        else:
            save_dict = arg_dict

        with open(file_name, write_mode) as f:
            json.dump(save_dict, f)

    def save(self, file_name: str, prefix: str = None, write_mode: str = "a"):
        """
        Save the system, loaded in by system.system.load
        """
        arg_dict = self.name_and_arguments_to_dict()
        self.argument_saver(arg_dict, file_name, prefix, write_mode)

    @staticmethod
    def argument_loader(file_name: str, prefix: str = None):
        """
        Load in the dictionary of arguments for system.system.from_dict
        """
        with open(file_name, "r") as f:
            load_dict = json.load(f)

        if not prefix == None:
            out_dict = load_dict[prefix]
        else:
            out_dict = load_dict

        return out_dict
