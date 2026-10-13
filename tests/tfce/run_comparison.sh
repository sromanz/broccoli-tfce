#!/bin/bash
# Run FSL randomise -T and BROCCOLI RandomiseGroupLevel (TFCE) on the same one-sample data,
# time both, and compare the outputs voxel by voxel.
#
# Usage:
#   tests/tfce/run_comparison.sh <data4D.nii.gz> <mask.nii.gz> <outdir> [n_perm] [-- extra BROCCOLI options]
#   tests/tfce/run_comparison.sh --synthetic <outdir> [n_perm] [-- extra BROCCOLI options]
#
# Examples:
#   tests/tfce/run_comparison.sh --synthetic /tmp/cmp
#   tests/tfce/run_comparison.sh all_subjects.nii.gz mask.nii.gz cmp_out 5000 -- -platform 0 -device 0
#
# With N <= 12 subjects and n_perm >= 2^N both programs use ALL 2^N sign flips, so the corrected
# p-values must agree voxel by voxel. With more subjects both draw random sign flips: TFCE maps
# still agree, corrected p-values only up to Monte Carlo error.
#
# Needs: FSL (randomise, fslmaths, fslnvols) in PATH, python3 with numpy + nibabel,
#        BROCCOLI built with tests/tfce/build_local.sh (or set BROCCOLI_BIN).
set -e
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
TESTS="$ROOT/tests/tfce"
BROCCOLI_BIN=${BROCCOLI_BIN:-$ROOT/build/RandomiseGroupLevel}
export BROCCOLI_DIR="$ROOT/"
export LD_LIBRARY_PATH="$ROOT/code/BROCCOLI_LIB/clBLASLinux:$LD_LIBRARY_PATH"

# ---- arguments
EXTRA=()
POS=()
while [ $# -gt 0 ]; do
    if [ "$1" = "--" ]; then shift; EXTRA=("$@"); break; fi
    POS+=("$1"); shift
done
if [ "${POS[0]}" = "--synthetic" ]; then
    OUT=${POS[1]:?usage: run_comparison.sh --synthetic <outdir> [n_perm]}
    NPERM=${POS[2]:-5000}
    mkdir -p "$OUT"
    python3 "$TESTS/make_synthetic.py" "$OUT/input"
    DATA="$OUT/input/data.nii.gz"; MASK_IN="$OUT/input/mask.nii.gz"
else
    DATA=${POS[0]:?usage: run_comparison.sh <data4D> <mask> <outdir> [n_perm]}
    MASK_IN=${POS[1]:?missing mask}
    OUT=${POS[2]:?missing output directory}
    NPERM=${POS[3]:-5000}
fi

# ---- checks
for tool in randomise fslmaths fslnvols; do
    command -v $tool >/dev/null || { echo "ERROR: $tool not found - load FSL first (source \$FSLDIR/etc/fslconf/fsl.sh)" >&2; exit 1; }
done
[ -x "$BROCCOLI_BIN" ] || { echo "ERROR: $BROCCOLI_BIN not found - run tests/tfce/build_local.sh build first (or set BROCCOLI_BIN)" >&2; exit 1; }
python3 -c "import numpy, nibabel" 2>/dev/null || { echo "ERROR: python3 needs numpy and nibabel (pip install numpy nibabel)" >&2; exit 1; }

mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
LOG="$OUT/comparison.log"
: > "$LOG"

# BROCCOLI only uses voxels whose mask value is exactly 1, FSL uses > 0: give both the same binary mask
fslmaths "$MASK_IN" -bin "$OUT/mask_bin" -odt float
MASK="$OUT/mask_bin.nii.gz"

N=$(fslnvols "$DATA")
MAXFLIPS=$(python3 -c "print(2**$N)")
EXHAUSTIVE=$(python3 -c "print(1 if $NPERM >= 2**$N else 0)")
if [ "$EXHAUSTIVE" = "1" ]; then
    MODE="exhaustive: all $MAXFLIPS sign flips in both programs -> corrected p-values should match exactly"
else
    MODE="random: $NPERM of $MAXFLIPS possible sign flips -> corrected p-values match only statistically"
fi
echo "Subjects: $N | permutations requested: $NPERM | $MODE" | tee -a "$LOG"

secs() { python3 -c "import time; print(time.time())"; }

# ---- FSL randomise (one-sample t-test with sign flipping, TFCE, raw stat images)
echo; echo "Running FSL randomise ..." | tee -a "$LOG"
t0=$(secs)
randomise -i "$DATA" -o "$OUT/fsl" -m "$MASK" -1 -T -R -n "$NPERM" >> "$LOG" 2>&1
t1=$(secs)
FSL_TIME=$(python3 -c "print(round($t1-$t0,1))")
echo "  randomise took $FSL_TIME s" | tee -a "$LOG"

# ---- BROCCOLI: warm-up run so that kernel compilation is not included in the timing
echo; echo "Running BROCCOLI RandomiseGroupLevel ${EXTRA[*]} ..." | tee -a "$LOG"
( cd "$OUT" && "$BROCCOLI_BIN" "$DATA" -groupmean -mask "$MASK" -permutations 10 -inferencemode 3 -output "$OUT/warmup" "${EXTRA[@]}" ) >> "$LOG" 2>&1 \
    || { echo "ERROR: BROCCOLI failed, see $LOG" >&2; tail -5 "$LOG" >&2; exit 1; }
rm -f "$OUT"/warmup_perm_*
grep -m1 "Running on OpenCL" "$LOG" | sed 's/^/  /'
t0=$(secs)
( cd "$OUT" && "$BROCCOLI_BIN" "$DATA" -groupmean -mask "$MASK" -permutations "$NPERM" -inferencemode 3 \
      -output "$OUT/broccoli" -writepermutationvalues "$OUT/broccoli_null.txt" "${EXTRA[@]}" ) >> "$LOG" 2>&1
t1=$(secs)
BROC_TIME=$(python3 -c "print(round($t1-$t0,1))")
echo "  RandomiseGroupLevel took $BROC_TIME s" | tee -a "$LOG"

# ---- compare
echo; echo "Comparison (FSL vs BROCCOLI):" | tee -a "$LOG"
FSL_CRIT=$(grep "Critical Value for:.*tfce_corrp_tstat1" "$LOG" | awk '{print $NF}')
BROC_CRIT=$(grep "Permutation threshold for group mean" "$LOG" | tail -1 | awk '{print $(NF)}')
echo "  95th percentile of max-TFCE null: FSL $FSL_CRIT, BROCCOLI $BROC_CRIT" | tee -a "$LOG"
python3 "$TESTS/compare_with_fsl.py" "$OUT/fsl" "$OUT/broccoli" "$MASK" 2>&1 | sed 's/^/  /' | tee -a "$LOG"
echo | tee -a "$LOG"
python3 -c "print('  Speed-up: %.1fx  (randomise %.1f s, BROCCOLI %.1f s)' % ($FSL_TIME/max($BROC_TIME,1e-3), $FSL_TIME, $BROC_TIME))" | tee -a "$LOG"
echo "Full log: $LOG"
