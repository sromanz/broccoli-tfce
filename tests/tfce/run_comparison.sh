#!/bin/bash
# Run FSL randomise -T and BROCCOLI RandomiseGroupLevel (TFCE) on the same data, time both, and
# compare the outputs voxel by voxel.
#
# Usage:
#   run_comparison.sh <data4D> <mask> <outdir> [options] [-- extra BROCCOLI options]
#   run_comparison.sh --synthetic <outdir> [--two-groups] [options] [-- extra BROCCOLI options]
#
# Options:
#   -n <n_perm>            permutations (default 5000; both programs cap at the number of unique ones)
#   --design <design.mat>  FSL design matrix   } GLM design (e.g. two groups + covariates);
#   --con <design.con>     FSL t-contrasts      } without them a one-sample (group mean) test is run
#   --own-perms            let BROCCOLI draw its own permutations (statistical comparison only)
#
# Examples:
#   run_comparison.sh --synthetic /tmp/cmp
#   run_comparison.sh --synthetic /tmp/cmp2 --two-groups -n 1000
#   run_comparison.sh logjac_all.nii.gz mask.nii.gz cmp_tbm --design design.mat --con design.con -- -platform 0 -device 0
#
# By default randomise writes its permutations (-P) and BROCCOLI re-uses exactly those, so both
# test against the same null distribution: t-maps, TFCE maps, null distributions and corrected
# p-values must then agree (up to float rounding), whatever the number of subjects or the design.
#
# Needs: FSL (randomise, fslmaths, fslnvols) in PATH, python3 with numpy + nibabel,
#        BROCCOLI built with tests/tfce/build_local.sh (or set BROCCOLI_BIN).
set -e
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
TESTS="$ROOT/tests/tfce"
BROCCOLI_BIN=${BROCCOLI_BIN:-$ROOT/build/RandomiseGroupLevel}
export BROCCOLI_DIR="$ROOT/"
export LD_LIBRARY_PATH="$ROOT/code/BROCCOLI_LIB/clBLASLinux:$LD_LIBRARY_PATH"

usage() { sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

# ---- arguments
EXTRA=(); POS=(); NPERM=5000; DESIGN=""; CON=""; SYNTH=0; TWOGROUPS=0; OWNPERMS=0
while [ $# -gt 0 ]; do
    case "$1" in
        --) shift; EXTRA=("$@"); break ;;
        -n) NPERM=$2; shift 2 ;;
        --design) DESIGN=$2; shift 2 ;;
        --con) CON=$2; shift 2 ;;
        --synthetic) SYNTH=1; shift ;;
        --two-groups) TWOGROUPS=1; shift ;;
        --own-perms) OWNPERMS=1; shift ;;
        -h|--help) usage ;;
        *) POS+=("$1"); shift ;;
    esac
done

if [ $SYNTH = 1 ]; then
    OUT=${POS[0]:?missing output directory}
    [ -n "${POS[1]}" ] && NPERM=${POS[1]}          # old form: --synthetic <outdir> <n_perm>
    mkdir -p "$OUT"
    if [ $TWOGROUPS = 1 ]; then
        python3 "$TESTS/make_synthetic.py" "$OUT/input" --two-groups --subjects 20 --amplitude 3
        DESIGN="$OUT/input/design.mat"; CON="$OUT/input/design.con"
    else
        python3 "$TESTS/make_synthetic.py" "$OUT/input"
    fi
    DATA="$OUT/input/data.nii.gz"; MASK_IN="$OUT/input/mask.nii.gz"
else
    [ ${#POS[@]} -ge 3 ] || usage
    DATA=${POS[0]}; MASK_IN=${POS[1]}; OUT=${POS[2]}
    [ -n "${POS[3]}" ] && NPERM=${POS[3]}          # old form: <data> <mask> <outdir> <n_perm>
fi
if { [ -n "$DESIGN" ] && [ -z "$CON" ]; } || { [ -z "$DESIGN" ] && [ -n "$CON" ]; }; then
    echo "ERROR: --design and --con must be given together" >&2; exit 1
fi

# ---- checks
for tool in randomise fslmaths fslnvols; do
    command -v $tool >/dev/null || { echo "ERROR: $tool not found - load FSL first (source \$FSLDIR/etc/fslconf/fsl.sh)" >&2; exit 1; }
done
[ -x "$BROCCOLI_BIN" ] || { echo "ERROR: $BROCCOLI_BIN not found - run tests/tfce/build_local.sh build first (or set BROCCOLI_BIN)" >&2; exit 1; }
python3 -c "import numpy, nibabel" 2>/dev/null || { echo "ERROR: python3 needs numpy and nibabel (pip install numpy nibabel)" >&2; exit 1; }

mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
abspath() { echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; }
DATA=$(abspath "$DATA")
LOG="$OUT/comparison.log"
: > "$LOG"

# BROCCOLI only uses voxels whose mask value is exactly 1, FSL uses > 0: give both the same binary mask
fslmaths "$MASK_IN" -bin "$OUT/mask_bin" -odt float
MASK="$OUT/mask_bin.nii.gz"
N=$(fslnvols "$DATA")

# ---- design
if [ -n "$DESIGN" ]; then
    DESIGN=$(abspath "$DESIGN"); CON=$(abspath "$CON")
    echo "Design: $(python3 "$TESTS/fsl_to_broccoli_design.py" "$DESIGN" "$CON" "$OUT/broccoli_design.txt" "$OUT/broccoli_contrasts.txt")" | tee -a "$LOG"
    NCON=$(grep NumContrasts "$OUT/broccoli_contrasts.txt" | awk '{print $2}')
    FSL_DESIGN=(-d "$DESIGN" -t "$CON")
    BROC_DESIGN=(-design "$OUT/broccoli_design.txt" -contrasts "$OUT/broccoli_contrasts.txt")
    KIND="GLM"
else
    NCON=1
    FSL_DESIGN=(-1)
    BROC_DESIGN=(-groupmean)
    KIND="one-sample (group mean, sign flipping)"
fi
echo "Subjects: $N | test: $KIND | contrasts: $NCON | permutations requested: $NPERM" | tee -a "$LOG"

secs() { python3 -c "import time; print(time.time())"; }

# ---- FSL randomise: TFCE, raw stat images (-R), null distributions (-N), permutations (-P)
echo; echo "Running FSL randomise ..." | tee -a "$LOG"
t0=$(secs)
randomise -i "$DATA" -o "$OUT/fsl" -m "$MASK" "${FSL_DESIGN[@]}" -T -R -N -P -n "$NPERM" >> "$LOG" 2>&1 \
    || { echo "ERROR: randomise failed, see $LOG" >&2; tail -5 "$LOG" >&2; exit 1; }
t1=$(secs)
FSL_TIME=$(python3 -c "print(round($t1-$t0,1))")
echo "  randomise took $FSL_TIME s" | tee -a "$LOG"

# ---- permutations for BROCCOLI: exactly the ones randomise used (one block per contrast)
PERM_OPTS=()
SHARED=0
if [ $OWNPERMS = 0 ]; then
    : > "$OUT/fsl_permutations_all.txt"
    ok=1
    for c in $(seq 1 $NCON); do
        f="$OUT/fsl_perm_tstat$c.txt"
        [ -s "$f" ] || { ok=0; break; }
        cat "$f" >> "$OUT/fsl_permutations_all.txt"
    done
    if [ $ok = 1 ]; then
        PERM_OPTS=(-permutationfile "$OUT/fsl_permutations_all.txt")
        SHARED=1
    else
        echo "  randomise did not write its permutations (-P); BROCCOLI will draw its own" | tee -a "$LOG"
    fi
fi

# ---- BROCCOLI: warm-up run so that GPU kernel compilation is not included in the timing
echo; echo "Running BROCCOLI RandomiseGroupLevel ${EXTRA[*]} ..." | tee -a "$LOG"
( cd "$OUT" && "$BROCCOLI_BIN" "$DATA" "${BROC_DESIGN[@]}" -mask "$MASK" -permutations 10 -inferencemode 3 -output "$OUT/warmup" "${EXTRA[@]}" ) >> "$LOG" 2>&1 \
    || { echo "ERROR: BROCCOLI failed, see $LOG" >&2; tail -5 "$LOG" >&2; exit 1; }
rm -f "$OUT"/warmup_perm_*
grep -m1 "Running on OpenCL" "$LOG" | sed 's/^/  /'

run_broccoli() {
    ( cd "$OUT" && "$BROCCOLI_BIN" "$DATA" "${BROC_DESIGN[@]}" -mask "$MASK" -permutations "$NPERM" -inferencemode 3 \
          -output "$OUT/broccoli" -writepermutationvalues "$OUT/broccoli_null.txt" "$@" "${EXTRA[@]}" ) >> "$LOG" 2>&1
}
t0=$(secs)
if ! run_broccoli "${PERM_OPTS[@]}"; then
    if [ $SHARED = 1 ]; then
        echo "  BROCCOLI could not use randomise's permutations (see log); retrying with its own" | tee -a "$LOG"
        grep -i "permutation file" "$LOG" | tail -1 | sed 's/^/    /'
        SHARED=0
        t0=$(secs)
        run_broccoli || { echo "ERROR: BROCCOLI failed, see $LOG" >&2; tail -5 "$LOG" >&2; exit 1; }
    else
        echo "ERROR: BROCCOLI failed, see $LOG" >&2; tail -5 "$LOG" >&2; exit 1
    fi
fi
t1=$(secs)
BROC_TIME=$(python3 -c "print(round($t1-$t0,1))")
echo "  RandomiseGroupLevel took $BROC_TIME s" | tee -a "$LOG"

# ---- compare
echo | tee -a "$LOG"
if [ $SHARED = 1 ]; then
    echo "Comparison (FSL vs BROCCOLI, same permutations -> should agree up to float rounding):" | tee -a "$LOG"
else
    echo "Comparison (FSL vs BROCCOLI, different random permutations -> p-values agree only statistically):" | tee -a "$LOG"
fi
for c in $(seq 1 $NCON); do
    [ $NCON -gt 1 ] && echo "  contrast $c: $(sed -n "$((c+3))p" "$OUT/broccoli_contrasts.txt" 2>/dev/null)" | tee -a "$LOG"
    python3 "$TESTS/compare_with_fsl.py" "$OUT/fsl" "$OUT/broccoli" "$MASK" --contrast $c \
        --fsl-null "$OUT/fsl_tfce_corrp_tstat$c.txt" --broccoli-null "$OUT/broccoli_null$c.txt" 2>&1 \
        | sed 's/^/    /' | tee -a "$LOG"
done
echo | tee -a "$LOG"
python3 -c "print('Speed-up: %.1fx  (randomise %.1f s, BROCCOLI %.1f s)' % ($FSL_TIME/max($BROC_TIME,1e-3), $FSL_TIME, $BROC_TIME))" | tee -a "$LOG"
echo "Full log: $LOG"
