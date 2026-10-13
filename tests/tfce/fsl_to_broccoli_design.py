#!/usr/bin/env python3
"""Convert FSL design files (VEST format: design.mat / design.con, e.g. from Glm or
design_ttest2) to BROCCOLI's RandomiseGroupLevel format.

Usage: fsl_to_broccoli_design.py design.mat design.con out_design.txt out_contrasts.txt

BROCCOLI format:
  design:     NumRegressors R / NumSubjects N / N rows of R values
  contrasts:  NumRegressors R / NumContrasts C / C rows of R values
BROCCOLI inverts X'X, so the design must have full column rank (e.g. use one column per group
without an extra intercept, as design_ttest2 does).
"""
import sys

import numpy as np


def read_vest(path):
    rows, in_matrix = [], False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("/Matrix"):
                in_matrix = True
                continue
            if line.startswith("/"):
                continue
            if in_matrix:
                rows.append([float(v) for v in line.split()])
    if not rows:
        sys.exit(f"{path}: no /Matrix section found - is this an FSL VEST file?")
    m = np.array(rows, dtype=float)
    return m


def main():
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    X = read_vest(sys.argv[1])
    C = np.atleast_2d(read_vest(sys.argv[2]))
    n, r = X.shape
    if C.shape[1] != r:
        sys.exit(f"design has {r} regressors but contrasts have {C.shape[1]} columns")
    rank = np.linalg.matrix_rank(X)
    if rank < r:
        sys.exit(f"design matrix is rank deficient (rank {rank} < {r} regressors); BROCCOLI needs a "
                 "full-rank design, e.g. drop the intercept when there is one column per group")
    if r > 25:
        sys.exit("BROCCOLI supports at most 25 regressors")
    fmt = lambda row: " ".join(repr(float(v)) for v in row)
    with open(sys.argv[3], "w") as f:
        f.write(f"NumRegressors {r}\nNumSubjects {n}\n\n")
        f.write("\n".join(fmt(row) for row in X) + "\n")
    with open(sys.argv[4], "w") as f:
        f.write(f"NumRegressors {r}\nNumContrasts {C.shape[0]}\n\n")
        f.write("\n".join(fmt(row) for row in C) + "\n")
    print(f"{n} subjects, {r} regressors, {C.shape[0]} contrasts")


if __name__ == "__main__":
    main()
