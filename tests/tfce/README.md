# TFCE in BROCCOLI — build, test and compare with FSL

GPU threshold-free cluster enhancement for `RandomiseGroupLevel`, reproducing FSL `randomise -T`.

## Usage

```
RandomiseGroupLevel data.nii.gz -groupmean -mask mask.nii.gz -inferencemode 3 -output res
RandomiseGroupLevel data.nii.gz -design design.mat -contrasts design.con -mask mask.nii.gz -inferencemode 3 -output res
```

| Option | Meaning | FSL equivalent |
|---|---|---|
| `-inferencemode 3` | TFCE, H = 2, E = 0.5, 6-connectivity | `-T` |
| `-T2` | TFCE for skeleton/2D data, H = 2, E = 1, 26-connectivity | `--T2` |
| `-tfceH`, `-tfceE`, `-tfceC` | height / extent exponent, connectivity (6, 18, 26) | `--tfce_H`, `--tfce_E`, `--tfce_C` |
| `-tfceD` | fixed step size (default max of unpermuted statistic / 100) | `--tfce_D` |

Outputs: `res_perm_tvalues.nii` (t-map), `res_perm_tfce.nii` (TFCE of the t-map, = FSL `_tfce_tstatN`),
`res_perm_pvalues.nii` (FWE-corrected 1-p, = FSL `_tfce_corrp_tstatN`).
F-tests with TFCE are not supported yet (FSL converts F to z first).

## Build (any OpenCL platform)

```
tests/tfce/build_local.sh build                       # -> build/RandomiseGroupLevel
export BROCCOLI_DIR=$PWD/ LD_LIBRARY_PATH=$PWD/code/BROCCOLI_LIB/clBLASLinux
```
On an NVIDIA machine the OpenCL headers/ICD loader come from the driver (`ocl-icd-opencl-dev`, `opencl-headers`);
set `OPENCL_INCLUDE` if the headers are not in `/usr/include/CL`. `clinfo -l` should list the Quadro.
Kernel binaries are cached in `compiled/Kernels/` and rebuilt automatically when kernel sources change.

## Self-test (no FSL needed)

```
python3 tests/tfce/tfce_ref.py                         # reference == literal port of FSL's tfce()
python3 tests/tfce/make_synthetic.py /tmp/syn
cd /tmp/syn
RandomiseGroupLevel data.nii.gz -groupmean -mask mask.nii.gz -permutations 300 -inferencemode 3 \
    -output tfce -writepermutationvalues permvalues.txt -writepermutations permvectors.txt
python3 <repo>/tests/tfce/validate.py /tmp/syn tfce   # prints PASS/FAIL
```

## Comparison with FSL randomise (exact)

With 12 subjects there are 2^12 = 4096 sign flips. randomise enumerates all of them when asked for
more, and BROCCOLI does the same with `-permutations 4096`, so both use the identical null set and
the corrected p-values should agree voxel by voxel.

```
randomise -i data.nii.gz -o fsl -m mask.nii.gz -1 -T -R -n 5000
RandomiseGroupLevel data.nii.gz -groupmean -mask mask.nii.gz -permutations 4096 -inferencemode 3 -output broc
python3 <repo>/tests/tfce/compare_with_fsl.py fsl broc mask.nii.gz
```
`-R` makes randomise write the raw `_tfce_tstat1` map. Expect tiny (~1e-6 relative) TFCE differences
from single- vs double-precision arithmetic.
