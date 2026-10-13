#!/usr/bin/env python3
"""Compare BROCCOLI TFCE outputs with FSL randomise outputs for the same data.

Usage:
  compare_with_fsl.py <fsl_prefix> <broccoli_prefix> <mask> [--contrast 1]

Expects <fsl_prefix>_tstatN, <fsl_prefix>_tfce_tstatN, <fsl_prefix>_tfce_corrp_tstatN (.nii.gz)
and <broccoli_prefix>_perm_tvalues.nii, _perm_tfce.nii, _perm_pvalues.nii.
"""
import argparse
import glob

import nibabel as nib
import numpy as np


def load(path_stem):
    hits = sorted(glob.glob(path_stem + ".nii*"))
    if not hits:
        raise SystemExit(f"missing {path_stem}.nii[.gz]")
    return np.asarray(nib.load(hits[0]).dataobj, np.float32)


def vol(arr, c):
    return arr[..., c] if arr.ndim == 4 else arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fsl_prefix")
    ap.add_argument("broccoli_prefix")
    ap.add_argument("mask")
    ap.add_argument("--contrast", type=int, default=1)
    a = ap.parse_args()
    c = a.contrast
    m = np.asarray(nib.load(a.mask).dataobj) > 0

    t_f = load(f"{a.fsl_prefix}_tstat{c}")
    tfce_f = load(f"{a.fsl_prefix}_tfce_tstat{c}")
    corrp_f = load(f"{a.fsl_prefix}_tfce_corrp_tstat{c}")
    t_b = vol(load(f"{a.broccoli_prefix}_perm_tvalues").squeeze(), c - 1)
    tfce_b = vol(load(f"{a.broccoli_prefix}_perm_tfce").squeeze(), c - 1)
    p_b = vol(load(f"{a.broccoli_prefix}_perm_pvalues").squeeze(), c - 1)

    dt = np.abs(t_f - t_b)[m]
    print(f"t-stat     : max |FSL - BROCCOLI| = {dt.max():.3g}")
    rel = np.abs(tfce_f - tfce_b)[m] / max(tfce_f.max(), 1e-12)
    print(f"TFCE       : FSL max {tfce_f.max():.6g}, BROCCOLI max {tfce_b.max():.6g}, "
          f"max rel. diff {rel.max():.3g}, voxels with rel. diff > 1e-4: {(rel > 1e-4).sum()}")
    dp = np.abs(corrp_f - p_b)[m]
    print(f"corrp (1-p): max |diff| = {dp.max():.3g}, voxels differing: {(dp > 1e-6).sum()} of {m.sum()}")
    for thr in (0.95, 0.99):
        sf, sb = (corrp_f[m] >= thr), (p_b[m] >= thr)
        print(f"  1-p >= {thr}: FSL {sf.sum()} voxels, BROCCOLI {sb.sum()} voxels, "
              f"agreement {np.mean(sf == sb) * 100:.3f}%")


if __name__ == "__main__":
    main()
