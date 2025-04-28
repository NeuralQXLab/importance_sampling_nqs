from einops import rearrange


def extract_patches1d(x, b):
    # This might not work, may need to add a batch dimension as in extract_patches2d
    # return rearrange(x, "(L_eff b) -> L_eff b", b=b)
    return  x.reshape(*x.shape[:-1], -1, b)


def extract_patches2d(x, b):
    """
    Extract bxb patches from the (nbatches, nsites) input x.
    Returns x reshaped to (nbatches,npatches,patch_size), where patch_size = b**2
    """
    batch = x.shape[0]
    L_eff = int((x.shape[1] // b**2) ** 0.5)
    x = x.reshape(batch, L_eff, b, L_eff, b)  # [L_eff, b, L_eff, b]
    x = x.transpose(0, 1, 3, 2, 4)  # [L_eff, L_eff, b, b]
    # flatten the patches
    x = x.reshape(batch, L_eff, L_eff, -1)  # [L_eff, L_eff, b*b]
    x = x.reshape(batch, L_eff * L_eff, -1)  # [L_eff*L_eff, b*b]
    return x
