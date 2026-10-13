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

Use this script, not BROCCOLI's old `compile_*.sh` scripts (those expect Intel/AMD SDKs under `/opt`).

```
# once, if not installed (the NVIDIA driver has the OpenCL runtime but not the headers):
sudo apt install opencl-headers ocl-icd-opencl-dev clinfo      # Fedora: opencl-headers ocl-icd-devel clinfo

tests/tfce/build_local.sh build                               # -> build/RandomiseGroupLevel
export BROCCOLI_DIR=$PWD/ LD_LIBRARY_PATH=$PWD/code/BROCCOLI_LIB/clBLASLinux:$LD_LIBRARY_PATH
```
The script finds `opencl.h` in `/usr/include/CL`, a CUDA install (`/usr/local/cuda*/include/CL`) or the
Intel/AMD SDK folders; otherwise set `OPENCL_INCLUDE=/path/to/CL` (and `OPENCL_LIB` for `libOpenCL.so`).
Without root, conda works: `conda install -c conda-forge khronos-opencl-icd-loader opencl-headers`.
`clinfo -l` should list the Quadro; pick it with `-platform P -device D` if it is not platform 0 / device 0.
Kernel binaries are cached in `compiled/Kernels/` and rebuilt automatically when kernel sources change.

## Running on the GPU

BROCCOLI runs on whatever OpenCL device you select with `-platform P -device D` (default 0 / 0).

1. List the OpenCL devices: `clinfo -l`. Platforms and devices are numbered from 0 in the order shown, e.g.
   ```
   Platform #0: NVIDIA CUDA
    `-- Device #0: Quadro P4000
   Platform #1: Portable Computing Language
    `-- Device #0: cpu-...
   ```
2. Pass the Quadro's numbers (not needed if it is platform 0, device 0):
   ```
   RandomiseGroupLevel data.nii.gz -groupmean -mask mask.nii.gz -inferencemode 3 -platform 0 -device 0 -output res
   ```
3. Check the first lines of the output: `Running on OpenCL platform 0 (Nvidia), device 0 (QuadroP4000)`.
   While it runs, `nvidia-smi` should list `RandomiseGroupLevel` as a process using the GPU.

If `clinfo -l` shows no NVIDIA platform although `nvidia-smi` works, the NVIDIA OpenCL driver part is missing:
check that `/etc/OpenCL/vendors/nvidia.icd` exists (Ubuntu: it comes with the `nvidia-opencl-icd` /
`libnvidia-compute-<version>` package matching your driver). In Docker, start the container with
`--gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility`.
The first run compiles the kernels for the GPU (can take a minute); later runs reuse the cached binaries.

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
