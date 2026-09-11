"""Landmark loading borrowed from Deep3DFaceRecon_pytorch.

https://github.com/sicxu/Deep3DFaceRecon_pytorch
"""

import numpy as np
from scipy.io import loadmat


def load_lm3d(mat_path=None):
    if mat_path is None:
        # Package-relative default so the loader works from any working directory.
        from importlib.resources import files

        mat_path = str(files("cupid.tddfa_v3") / "assets" / "similarity_Lm3D_all.mat")
    Lm3D = loadmat(mat_path)
    Lm3D = Lm3D['lm']

    # calculate 5 facial landmarks using 68 landmarks
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    Lm3D = np.stack([Lm3D[lm_idx[0], :], np.mean(Lm3D[lm_idx[[1, 2]], :], 0), np.mean(
        Lm3D[lm_idx[[3, 4]], :], 0), Lm3D[lm_idx[5], :], Lm3D[lm_idx[6], :]], axis=0)
    Lm3D = Lm3D[[1, 2, 0, 3, 4], :]

    return Lm3D
