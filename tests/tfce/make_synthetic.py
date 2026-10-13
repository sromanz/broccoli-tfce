#!/usr/bin/env python3
"""Generate a small synthetic group dataset for TFCE testing.

Writes <outdir>/data.nii.gz (4D, one volume per subject), <outdir>/mask.nii.gz
and <outdir>/design.mat / design.con (FSL VEST format).

Default: one-sample design (all subjects share a positive effect; design = column of ones).
--two-groups: like a TBM study. Group A has local "expansion" in one blob and "shrinkage" in
another relative to group B; an age covariate has its own effect in a third region.
design.mat = [groupA groupB age(demeaned)], design.con = [1 -1 0] (A > B) and [-1 1 0] (A < B).
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
            f.write(" ".join(f"{v:.6g}" for v in row) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--subjects", type=int, default=12)
    ap.add_argument("--shape", type=int, nargs=3, default=[40, 48, 40])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--amplitude", type=float, default=1.5, help="scale of the planted effect")
    ap.add_argument("--two-groups", action="store_true",
                    help="two-group design with an age covariate (first half = group A)")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    X, Y, Z = a.shape
    shape_zyx = (Z, Y, X)
    N = a.subjects

    if a.two_groups:
        nA = N // 2
        group_a = np.zeros(N)
        group_a[:nA] = 1
        age = rng.uniform(55, 85, N)
        age_c = age - age.mean()
        expansion = blob(shape_zyx, (20, 24, 12), 3.0)
        shrinkage = blob(shape_zyx, (18, 20, 28), 4.0)
        age_effect = blob(shape_zyx, (12, 32, 20), 3.0)
        signals = [a.amplitude * (g * expansion - 0.8 * g * shrinkage) + 0.05 * ac * age_effect
                   for g, ac in zip(group_a, age_c)]
        design = np.column_stack([group_a, 1 - group_a, age_c])
        contrasts = np.array([[1, -1, 0], [-1, 1, 0]])
    else:
        common = a.amplitude * (1.0 * blob(shape_zyx, (20, 24, 12), 3.0) + 0.6 * blob(shape_zyx, (18, 20, 28), 4.0))
        signals = [common] * N
        design = np.ones((N, 1))
        contrasts = np.ones((1, 1))

    vols = []
    for s in range(N):
        noise = ndimage.gaussian_filter(rng.standard_normal(shape_zyx), 1.5)
        noise /= noise.std()
        vols.append(signals[s] + noise)
    data_zyx = np.stack(vols, axis=-1)                      # (Z,Y,X,N)
    data = np.transpose(data_zyx, (2, 1, 0, 3)).astype(np.float32)  # (X,Y,Z,N)

    mask = np.zeros((X, Y, Z), np.float32)
    mask[2:-2, 2:-2, 2:-2] = 1.0
    data *= mask[..., None]

    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    nib.save(nib.Nifti1Image(data, aff), os.path.join(a.outdir, "data.nii.gz"))
    nib.save(nib.Nifti1Image(mask, aff), os.path.join(a.outdir, "mask.nii.gz"))
    write_vest(os.path.join(a.outdir, "design.mat"), design)
    write_vest(os.path.join(a.outdir, "design.con"), contrasts)


if __name__ == "__main__":
    main()
