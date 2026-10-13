#!/usr/bin/env python3
"""Validate BROCCOLI GPU TFCE (RandomiseGroupLevel -groupmean -inferencemode 3) against the
FSL-compatible reference in tfce_ref.py.

Usage: validate.py <dir> <output_prefix> [--H 2 --E 0.5 --C 6]
<dir> must contain data.nii.gz, mask.nii.gz and the BROCCOLI outputs
<prefix>_perm_tvalues, <prefix>_perm_tfce, <prefix>_perm_pvalues (.nii or .nii.gz),
permvalues1.txt (-writepermutationvalues permvalues.txt) and
permvectors.txt (-writepermutations permvectors.txt).
"""
import argparse
import glob
import os
import sys

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tfce_ref import randomise_delta, tfce_fsl  # noqa: E402


def one_sample_t(data, signs):
    x = data * signs[None, None, None, :]
    n = x.shape[-1]
    m = x.mean(-1)
    sd = x.std(-1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = m / (sd / np.sqrt(n))
    return np.nan_to_num(t).astype(np.float32)


def fsl_corrp(observed, null):
    """FSL OutputStat: 1-p = #(null maxima strictly below observed) / nPerms."""
    null = np.sort(np.asarray(null, np.float32))
    return np.searchsorted(null, observed.astype(np.float32), side="left") / float(len(null))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("prefix")
    ap.add_argument("--H", type=float, default=2.0)
    ap.add_argument("--E", type=float, default=0.5)
    ap.add_argument("--C", type=int, default=6)
    ap.add_argument("--nperm-check", type=int, default=50, help="permutations to recompute")
    a = ap.parse_args()
    d = lambda f: os.path.join(a.dir, f)

    def out(suffix):
        hits = sorted(glob.glob(d(a.prefix + suffix + ".nii*")) + glob.glob(d(a.prefix + suffix + ".hdr*")))
        if not hits:
            sys.exit(f"missing {a.prefix}{suffix}.nii[.gz]")
        return np.asarray(nib.load(hits[0]).dataobj, np.float32).squeeze()

    data = nib.load(d("data.nii.gz")).get_fdata(dtype=np.float32)
    mask = nib.load(d("mask.nii.gz")).get_fdata(dtype=np.float32) == 1
    t_b = out("_perm_tvalues")
    tfce_b = out("_perm_tfce")
    p_b = out("_perm_pvalues")
    null_b = np.loadtxt(d("permvalues1.txt"), dtype=np.float32).ravel()
    signs = np.loadtxt(d("permvectors.txt"), dtype=np.float32)
    ok = True

    # 1) t-maps: BROCCOLI vs numpy
    t_ref = one_sample_t(data, signs[0]) * mask
    print(f"[t-map] max |BROCCOLI - numpy| = {np.abs(t_b - t_ref)[mask].max():.3g}")

    # 2) observed TFCE: reference applied to BROCCOLI's own t-map, FSL step size
    delta = randomise_delta(t_b[mask])
    ref = tfce_fsl(np.where(mask, t_b, 0), delta=delta, H=a.H, E=a.E, connectivity=a.C)
    rel = np.abs(tfce_b - ref)[mask].max() / ref.max()
    print(f"[observed TFCE] delta = {delta:.6g}, max = {tfce_b.max():.6g} (ref {ref.max():.6g}), "
          f"max rel. error = {rel:.3g}")
    ok &= rel < 1e-5

    # 3) null distribution: recompute a subset of permutations independently
    k = min(a.nperm_check, len(signs))
    rels = []
    for i in range(k):
        t = one_sample_t(data, signs[i]) * mask
        m = tfce_fsl(t, delta=delta, H=a.H, E=a.E, connectivity=a.C).max()
        rels.append(abs(m - null_b[i]) / max(m, 1e-12))
    rels = np.array(rels)
    print(f"[null] permutations checked = {k}, max rel. error of max-TFCE = {rels.max():.3g}, "
          f"median = {np.median(rels):.3g}")
    ok &= rels.max() < 1e-4

    # 4) corrected p-values with FSL's rule on BROCCOLI's own null and observed map
    corrp = fsl_corrp(tfce_b[mask], null_b)
    diff = np.abs(corrp - p_b[mask]).max()
    print(f"[corrp] max |BROCCOLI - FSL rule| = {diff:.3g}; voxels with 1-p >= 0.95: {(p_b[mask] >= 0.95).sum()}")
    ok &= diff < 1e-6

    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
