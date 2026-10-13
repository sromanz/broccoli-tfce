#!/bin/bash
# Build the BROCCOLI library + RandomiseGroupLevel against the system OpenCL (any vendor).
# Usage: tests/tfce/build_local.sh [build_dir]
# Optional environment variables:
#   OPENCL_INCLUDE  directory that contains opencl.h        (auto-detected)
#   OPENCL_LIB      directory that contains libOpenCL.so    (auto-detected)
set -e
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
B=${1:-$ROOT/build}
mkdir -p "$B"
# The script changes directory below, so turn a relative build path into an absolute one
B=$(cd "$B" && pwd)

# ---- locate opencl.h (BROCCOLI includes it as <opencl.h>, so we need the CL/ directory itself)
find_cl_include() {
    local candidates=(
        "$OPENCL_INCLUDE"
        "$OPENCL_INCLUDE/CL"
        /usr/include/CL
        /usr/local/include/CL
        ${CUDA_HOME:+$CUDA_HOME/include/CL}
        ${CUDA_PATH:+$CUDA_PATH/include/CL}
        /usr/local/cuda/include/CL
        /usr/local/cuda-*/include/CL
        /opt/cuda/include/CL
        /usr/lib/cuda/include/CL
        /opt/intel/opencl-sdk/include/CL
        /opt/AMDAPP/include/CL
    )
    for d in "${candidates[@]}"; do
        [ -n "$d" ] && [ -f "$d/opencl.h" ] && { echo "$d"; return 0; }
    done
    return 1
}

if ! INC_CL=$(find_cl_include); then
    cat >&2 <<'MSG'
ERROR: opencl.h not found.

The OpenCL headers are not installed (the NVIDIA driver provides the OpenCL runtime,
but not the headers). Install them with one of:

  Ubuntu/Debian : sudo apt install opencl-headers ocl-icd-opencl-dev clinfo
  Fedora/RHEL   : sudo dnf install opencl-headers ocl-icd-devel clinfo
  conda (no root): conda install -c conda-forge khronos-opencl-icd-loader opencl-headers
                   then: OPENCL_INCLUDE=$CONDA_PREFIX/include/CL OPENCL_LIB=$CONDA_PREFIX/lib tests/tfce/build_local.sh

If you know where opencl.h is, run:  OPENCL_INCLUDE=/path/to/CL tests/tfce/build_local.sh
MSG
    exit 1
fi
echo "Using OpenCL headers in $INC_CL"

# ---- locate libOpenCL.so (only needed if it is not in a default linker path)
LIB_FLAGS=""
if [ -n "$OPENCL_LIB" ]; then
    LIB_FLAGS="-L$OPENCL_LIB -Wl,-rpath,$OPENCL_LIB"
elif ! ldconfig -p 2>/dev/null | grep -q "libOpenCL.so\b"; then
    for d in /usr/local/cuda/lib64 /usr/local/cuda-*/lib64 /opt/cuda/lib64 ${CUDA_HOME:+$CUDA_HOME/lib64}; do
        if ls "$d"/libOpenCL.so* >/dev/null 2>&1; then
            LIB_FLAGS="-L$d -Wl,-rpath,$d"
            echo "Using libOpenCL from $d"
            break
        fi
    done
fi

# Not all header packages ship the unversioned libOpenCL.so symlink needed for linking
if [ -z "$LIB_FLAGS" ] && ! ldconfig -p 2>/dev/null | grep -q "libOpenCL.so =>"; then
    if ldconfig -p 2>/dev/null | grep -q "libOpenCL.so.1"; then
        echo "WARNING: only libOpenCL.so.1 found; if linking fails install ocl-icd-opencl-dev (or ocl-icd-devel)." >&2
    else
        echo "WARNING: libOpenCL not found; install ocl-icd-opencl-dev (or set OPENCL_LIB)." >&2
    fi
fi

CL_FLAGS="-I$INC_CL -I$(dirname "$INC_CL") -DCL_TARGET_OPENCL_VERSION=120 -DCL_USE_DEPRECATED_OPENCL_1_2_APIS"

echo "Compiling BROCCOLI library (takes ~30 s)..."
cd "$ROOT/code/BROCCOLI_LIB"
g++ $CL_FLAGS -I. -IEigen -IclBLASLinux -O2 -DNDEBUG -m64 -fopenmp -fPIC -w -c -o "$B/broccoli_lib.o" broccoli_lib.cpp
ar rcs "$B/libBROCCOLI_LIB.a" "$B/broccoli_lib.o"

echo "Linking RandomiseGroupLevel..."
cd "$ROOT/code/Bash_Wrapper"
g++ -no-pie RandomiseGroupLevel.cpp $CL_FLAGS -I../BROCCOLI_LIB -I../BROCCOLI_LIB/Eigen -I../BROCCOLI_LIB/clBLASLinux \
  -Inifticlib-2.0.0/niftilib -Inifticlib-2.0.0/znzlib -L"$B" -L../BROCCOLI_LIB/clBLASLinux -Lnifticlib-2.0.0/lib $LIB_FLAGS \
  -lBROCCOLI_LIB -lOpenCL -lclBLAS -lniftiio -lznz -lz -O2 -fopenmp -w -o "$B/RandomiseGroupLevel"

echo
echo "Built $B/RandomiseGroupLevel"
echo "Before running:"
echo "  export BROCCOLI_DIR=$ROOT/ LD_LIBRARY_PATH=$ROOT/code/BROCCOLI_LIB/clBLASLinux:\$LD_LIBRARY_PATH"
if command -v clinfo >/dev/null 2>&1; then
    echo "OpenCL devices on this machine (use -platform P -device D to choose):"
    clinfo -l 2>/dev/null | sed 's/^/  /'
fi
