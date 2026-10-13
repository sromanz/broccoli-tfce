#!/usr/bin/env python3
"""Reference TFCE matching FSL randomise / newimage `tfce()` semantics.

Behaviour reproduced from FSL source (newimage/newimagefns.h `tfce`, randomise.cc
`tfceStatistic`):
  * Only voxels with value > 0 take part; the outermost voxel layer of the volume
    is ignored (FSL loops x in [1, xsize-2] etc.).
  * Thresholds h = 0, d, 2d, ... while h < max + d, accumulated in float32.
  * A voxel belongs to the suprathreshold set at h if value > h (strict).
  * Each step adds  size^E * h^H  to every voxel of a cluster -- note: NOT multiplied by d.
  * Cluster size is a voxel count; connectivity 6 / 18 / 26.
  * randomise: d = max(unpermuted stat)/100, computed ONCE per contrast from the
    unpermuted map and reused for every permutation (unless --tfce_D is given).
    Per-map max is still used for the upper integration limit.
  * F-statistics are converted to z before TFCE (not handled here).
Defaults: -T  -> H=2, E=0.5, C=6 ;  --T2 -> H=2, E=1, C=26.
"""
from collections import deque

import numpy as np
from scipy import ndimage


def _structure(connectivity):
    rank = {6: 1, 18: 2, 26: 3}[connectivity]
    return ndimage.generate_binary_structure(3, rank)


def _thresholds(max_t, delta):
    """Exactly FSL's float loop: for(float h=0; h < maxT+delta; h += delta)."""
    max_t = np.float32(max_t)
    delta = np.float32(delta)
    limit = np.float32(max_t + delta)
    h = np.float32(0.0)
    out = []
    while h < limit:
        out.append(h)
        h = np.float32(h + delta)
    return out


def tfce_fsl(vol, delta=0.0, H=2.0, E=0.5, connectivity=6):
    """TFCE of a 3D float volume (any axis order), FSL-compatible. Returns float32."""
    vol = np.asarray(vol, dtype=np.float32)
    max_t = np.float32(vol.max())
    if delta == 0:
        delta = np.float32(max_t / np.float32(100.0))
    if delta <= 0:
        raise ValueError("tfce requires a positive delta")
    interior = np.zeros(vol.shape, bool)
    interior[1:-1, 1:-1, 1:-1] = True
    struct = _structure(connectivity)
    H32 = np.float32(H)
    acc = np.zeros(vol.shape, np.float32)
    for h in _thresholds(max_t, delta):
        supra = interior & (vol > h)
        if not supra.any():
            continue
        labels, n = ndimage.label(supra, structure=struct)
        sizes = np.bincount(labels.ravel(), minlength=n + 1).astype(np.float64)
        hh = np.float64(np.power(h, H32, dtype=np.float32))
        contrib = np.power(sizes, np.float64(E)) * hh
        contrib[0] = 0.0
        vals = contrib[labels]
        acc = np.where(supra, (acc.astype(np.float64) + vals).astype(np.float32), acc)
    return acc


def tfce_fsl_literal(vol, delta=0.0, H=2.0, E=0.5, connectivity=6):
    """Slow, line-by-line port of FSL's flood-fill implementation (for tiny volumes)."""
    vol = np.asarray(vol, dtype=np.float32)
    X, Y, Z = vol.shape
    max_t = np.float32(vol.max())
    if delta == 0:
        delta = np.float32(max_t / np.float32(100.0))
    offs = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                d2 = dx * dx + dy * dy + dz * dz
                need = {0: 100, 1: 6, 2: 18, 3: 26}[d2]
                if connectivity >= need:
                    offs.append((dx, dy, dz))
    enh = np.zeros(vol.shape, np.float32)
    pts = [(x, y, z) for z in range(1, Z - 1) for y in range(1, Y - 1) for x in range(1, X - 1)
           if vol[x, y, z] > 0]
    for h in _thresholds(max_t, delta):
        act = {p for p in pts if vol[p] > h}
        lab = {}
        sizes = []
        for p in act:
            if p in lab:
                continue
            lab[p] = len(sizes)
            q = deque([p])
            n = 1
            while q:
                x, y, z = q.popleft()
                for dx, dy, dz in offs:
                    r = (x + dx, y + dy, z + dz)
                    if r in act and r not in lab:
                        lab[r] = lab[p]
                        q.append(r)
                        n += 1
            sizes.append(n)
        hh = np.float64(np.power(h, np.float32(H), dtype=np.float32))
        for p, l in lab.items():
            enh[p] = np.float32(np.float64(enh[p]) + (sizes[l] ** float(E)) * hh)
    return enh


def randomise_delta(unpermuted_stat_in_mask):
    """Delta randomise uses for every permutation of one contrast:
    float(double(max) / 100.0), from the unpermuted statistic inside the mask."""
    return np.float32(np.float64(np.float32(np.max(unpermuted_stat_in_mask))) / 100.0)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for conn in (6, 18, 26):
        v = ndimage.gaussian_filter(rng.standard_normal((9, 10, 8)), 1.0).astype(np.float32) * 4
        a = tfce_fsl(v, connectivity=conn)
        b = tfce_fsl_literal(v, connectivity=conn)
        print(conn, "max", a.max(), "max abs diff", np.abs(a - b).max())
        assert np.allclose(a, b, rtol=1e-6, atol=1e-6)
    print("fast reference == literal FSL port")
