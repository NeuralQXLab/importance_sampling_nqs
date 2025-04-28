from .body import Embed, Encoder
from .heads import OutputHead, LayerSum, RBM_noLayer, noLayer
from flax import linen as nn
from typing import Callable


class ViT(nn.Module):
    num_layers: int
    d_model: int
    heads: int
    L_eff: int
    """
        Total number of patches
    """
    b: int
    """
        The size of a patch along each linear dimension
    """
    extract_patches: Callable
    """
        Function for patchifying input
    """
    Head: nn.module
    """
        Output head after encoder
    """
    expansion_factor: int = 4
    """
        Factor to expand model dimension in feedforward block 
    """
    transl_invariant: bool = True
    two_dimensional: bool = False

    def setup(self):
        self.patches_and_embed = Embed(
            self.d_model, self.b, extract_patches=self.extract_patches
        )

        self.encoder = Encoder(
            num_layers=self.num_layers,
            d_model=self.d_model,
            h=self.heads,
            L_eff=self.L_eff,
            expansion_factor=self.expansion_factor,
            transl_invariant=self.transl_invariant,
            two_dimensional=self.two_dimensional,
        )

        self.output = self.Head(self.d_model)

    def __call__(self, x):
        x = self.patches_and_embed(x)

        x = self.encoder(x)

        z = self.output(x)

        return z


def ViT_Vanilla(
    num_layers: int,
    d_model: int,
    heads: int,
    L_eff: int,
    b: int,
    extract_patches: Callable,
    expansion_factor: int = 4,
    transl_invariant: bool = True,
    two_dimensional: bool = False,
):
    return ViT(
        num_layers,
        d_model,
        heads,
        L_eff,
        b,
        extract_patches,
        OutputHead,
        expansion_factor,
        transl_invariant,
        two_dimensional,
    )


def ViT_LayerSum(
    num_layers: int,
    d_model: int,
    heads: int,
    L_eff: int,
    b: int,
    extract_patches: Callable,
    expansion_factor: int = 4,
    transl_invariant: bool = True,
    two_dimensional: bool = False,
):
    return ViT(
        num_layers,
        d_model,
        heads,
        L_eff,
        b,
        extract_patches,
        LayerSum,
        expansion_factor,
        transl_invariant,
        two_dimensional,
    )


def ViT_RBMnoLayer(
    num_layers: int,
    d_model: int,
    heads: int,
    L_eff: int,
    b: int,
    extract_patches: Callable,
    expansion_factor: int = 4,
    transl_invariant: bool = True,
    two_dimensional: bool = False,
):
    return ViT(
        num_layers,
        d_model,
        heads,
        L_eff,
        b,
        extract_patches,
        RBM_noLayer,
        expansion_factor,
        transl_invariant,
        two_dimensional,
    )


def ViT_noLayer(
    num_layers: int,
    d_model: int,
    heads: int,
    L_eff: int,
    b: int,
    extract_patches: Callable,
    expansion_factor: int = 4,
    transl_invariant: bool = True,
    two_dimensional: bool = False,
):
    return ViT(
        num_layers,
        d_model,
        heads,
        L_eff,
        b,
        extract_patches,
        noLayer,
        expansion_factor,
        transl_invariant,
        two_dimensional,
    )
