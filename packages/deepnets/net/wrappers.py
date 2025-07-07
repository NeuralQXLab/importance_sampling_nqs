# Wrap the neural networks to define parameters and save
from .CNN import ConvReLU, final_actfn
from .base_wrapper import NetBase
import argparse

# import deepnets.system as system
import json
from deepnets.net import ViT


class ResCNN(NetBase):
    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--depth", type=int, required=True, help="Number of CNN layers"
        )
        parser.add_argument(
            "--features",
            type=int,
            required=True,
            help="Number of features in each layer (fixed)",
        )
        parser.add_argument(
            "--kernel_width",
            type=int,
            required=True,
            help="Width of the convolutional kernel",
        )

    @staticmethod
    def read_arguments(args: argparse.Namespace):
        return args.depth, args.features, args.kernel_width

    def __init__(self, depth: int, features: int, kernel_width: int, system):
        self.name = "ResCNN"
        self.depth = depth
        self.features = features
        self.kernel_width = kernel_width
        self.network = ConvReLU(
            depth=self.depth,
            features=self.features,
            kernel_size=(self.kernel_width, self.kernel_width),
            graph=system.graph,
            final_actfn=final_actfn,
        )

    def name_and_arguments_to_dict(self):
        """
        Convert the name and arguments for __init__ (except system) to a dictionary
        """
        arg_dict = {
            "name": self.name,
            "depth": self.depth,
            "features": self.features,
            "kernel_width": self.kernel_width,
        }
        return arg_dict


class ViT2D(NetBase):
    nets = {
        "Vanilla": ViT.Vanilla,
        "LayerSum": ViT.LayerSum,
        "RBMnoLayer": ViT.RBMnoLayer,
        "noLayer": ViT.noLayer,
    }

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--depth", type=int, required=True, help="Number of encoder layers"
        )
        parser.add_argument(
            "--d_model",
            type=int,
            required=True,
            help="Model dimension (number of features)",
        )
        parser.add_argument(
            "--heads",
            type=int,
            required=True,
            help="Number of heads in attention mechanism",
        )
        parser.add_argument(
            "--linear_patch_size",
            type=int,
            required=True,
            help="Linear size of a patch",
        )
        parser.add_argument(
            "--output_head",
            type=str,
            required=True,
            help="Which output head to use (Vanilla / LayerSum / RBMnoLayer / noLayer)",
        )
        parser.add_argument(
            "--expansion_factor",
            type=int,
            required=True,
            help="Factor to expand model dimension in feedforward block",
        )

    @staticmethod
    def read_arguments(args: argparse.Namespace):
        return (
            args.depth,
            args.d_model,
            args.heads,
            args.linear_patch_size,
            args.output_head,
            args.expansion_factor,
        )

    def __init__(
        self,
        depth: int,
        d_model: int,
        heads: int,
        linear_patch_size: int,
        output_head_name: str,
        expansion_factor: int,
        system,
    ):
        self.name = "ViT2D"
        self.depth = depth
        self.d_model = d_model
        self.heads = heads
        self.linear_patch_size = linear_patch_size
        self.output_head_name = output_head_name
        self.b = linear_patch_size  # size of patch along linear dimension
        self.Np = system.L**2 // self.b**2
        self.expansion_factor = expansion_factor
        self.network = self.nets[self.output_head_name](
            num_layers=self.depth,
            d_model=self.d_model,
            heads=self.heads,
            L_eff=self.Np,
            b=self.b,
            extract_patches=system.extract_patches_as1d,
            expansion_factor=expansion_factor,
            transl_invariant=True,
            two_dimensional=True,
        )

    def name_and_arguments_to_dict(self):
        """
        Convert the arguments for __init__ (except system) to a dictionary
        """
        arg_dict = {
            "name": self.name,
            "depth": self.depth,
            "d_model": self.d_model,
            "heads": self.heads,
            "linear_patch_size": self.linear_patch_size,
            "output_head_name": self.output_head_name,
            "expansion_factor": self.expansion_factor,
        }
        return arg_dict


networks = {
    "ResCNN": ResCNN,
    "ViT2D": ViT2D,
}


def from_dict(arg_dict: dict, system, network_name="ConvNext"):
    """
    Return the wrapped network specified by the dictionary
    """
    try:
        network = networks[str(arg_dict["name"])]
        del arg_dict["name"]
    except KeyError:  # compatibility with old versions where it wasnt saved
        network = networks[network_name]
        arg_dict["net_type"] = arg_dict["output_head"]
        del arg_dict["output_head"]
        arg_dict["init_kernel_width"] = 1
    try:  # stupid fix for these being saved as lists
        arg_dict["n_blocks"] = tuple(arg_dict["n_blocks"])
        arg_dict["features"] = tuple(arg_dict["features"])
    except KeyError:
        pass
    # print(arg_dict)
    return network(**arg_dict, system=system)


def load(file_name: str, system, prefix: str = None):
    """
    Return the wrapped network specified by the dictionary, dict[prefix], contained in
    the json file file_filename
    """
    arg_dict = NetBase.argument_loader(file_name, prefix)
    loaded_network = from_dict(arg_dict, system)
    return loaded_network
