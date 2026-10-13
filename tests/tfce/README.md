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

Outputs: `res_perm_tvalues.nii.gz` (t-map), `res_perm_tfce.nii.gz` (TFCE of the t-map, = FSL `_tfce_tstatN`),
`res_perm_pvalues.nii.gz` (FWE-corrected 1-p, = FSL `_tfce_corrp_tstatN`).
Like FSL, outputs are compressed `.nii.gz` unless `FSLOUTPUTTYPE=NIFTI` is set (then `.nii`);
an explicit extension on `-output` (e.g. `-output res.nii`) is always kept.
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

## Comparison with FSL randomise

`run_comparison.sh` runs both programs on the same data with the same binary mask, times them
and compares t-maps, TFCE maps and FWE-corrected p-values voxel by voxel (FSL must be loaded):

```
tests/tfce/run_comparison.sh --synthetic /tmp/cmp                                   # 12-subject test data
tests/tfce/run_comparison.sh all_subjects.nii.gz mask.nii.gz cmp_out 5000 -- -platform 0 -device 0
```
It runs `randomise -i data -o fsl -m mask -1 -T -R -n <n_perm>` and
`RandomiseGroupLevel data -groupmean -mask mask -inferencemode 3 -permutations <n_perm>`
(after an untimed warm-up run, so GPU kernel compilation is not counted), then prints a report and
writes everything to `<outdir>/comparison.log`.

What to expect:
- **N <= 12 subjects** (n_perm >= 2^N): both programs use all 2^N sign flips, i.e. the identical null
  distribution, so the corrected p-values should be identical in every voxel and the 95th percentiles
  of the null should agree to ~1e-6 relative.
- **More subjects**: both draw random sign flips. t- and TFCE maps still agree (~1e-6 relative; a
  handful of voxels can differ more when a value lies within float rounding of a TFCE step), corrected
  p-values agree within Monte Carlo error (about ±0.006 at p = 0.05 with 5000 permutations).
- The script compares one-sample (group mean) designs. For GLM designs, run both programs by hand;
  `compare_with_fsl.py fsl broccoli mask.nii.gz --contrast N` compares any contrast.
