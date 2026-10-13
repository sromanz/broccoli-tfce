#!/usr/bin/env python3
"""Generate a small synthetic group dataset for TFCE testing.

Writes <outdir>/data.nii.gz (4D, one volume per subject), <outdir>/mask.nii.gz
and <outdir>/design.mat / design.con (FSL vest format, one-sample design).
"""
import argparse
import os

import nibabel as nib
import numpy as np
from scipy import ndimage


def blob(shape, centre, sigma):
    zz, yy, xx = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    d2 = (zz - centre[0]) ** 2 + (yy - centre[1]) ** 2 + (xx - centre[2]) ** 2
    return np.exp(-d2 / (2 * sigma**2))


def write_vest(path, mat):
    mat = np.atleast_2d(mat)
    with open(path, "w") as f:
        f.write(f"/NumWaves\t{mat.shape[1]}\n/NumPoints\t{mat.shape[0]}\n/Matrix\n")
        for row in mat:
            f.write(" ".join(f"{v:g}" for v in row) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--subjects", type=int, default=12)
    ap.add_argument("--shape", type=int, nargs=3, default=[40, 48, 40])
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    X, Y, Z = a.shape
    shape_zyx = (Z, Y, X)

    signal = 0.9 * blob(shape_zyx, (20, 24, 12), 3.0) + 0.6 * blob(shape_zyx, (18, 20, 28), 4.0)
    vols = []
    for _ in range(a.subjects):
        noise = ndimage.gaussian_filter(rng.standard_normal(shape_zyx), 1.5)
        noise /= noise.std()
        vols.append(signal + noise)
    data_zyx = np.stack(vols, axis=-1)                      # (Z,Y,X,N)
    data = np.transpose(data_zyx, (2, 1, 0, 3)).astype(np.float32)  # (X,Y,Z,N)

    mask = np.zeros((X, Y, Z), np.float32)
    mask[2:-2, 2:-2, 2:-2] = 1.0
    data *= mask[..., None]

    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    nib.save(nib.Nifti1Image(data, aff), os.path.join(a.outdir, "data.nii.gz"))
    nib.save(nib.Nifti1Image(mask, aff), os.path.join(a.outdir, "mask.nii.gz"))
    write_vest(os.path.join(a.outdir, "design.mat"), np.ones((a.subjects, 1)))
    write_vest(os.path.join(a.outdir, "design.con"), np.ones((1, 1)))


if __name__ == "__main__":
    main()
