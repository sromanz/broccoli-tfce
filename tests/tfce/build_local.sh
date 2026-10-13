#!/bin/bash
# Build BROCCOLI library + RandomiseGroupLevel against the system OpenCL (any vendor).
# Usage: tests/tfce/build_local.sh <build_dir>
set -e
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
B=${1:-$ROOT/build}
mkdir -p "$B"
INC_CL=${OPENCL_INCLUDE:-/usr/include/CL}
cd "$ROOT/code/BROCCOLI_LIB"
g++ -I"$INC_CL" -I. -IEigen -IclBLASLinux -DCL_TARGET_OPENCL_VERSION=120 -O2 -DNDEBUG -m64 -fopenmp -fPIC -w -c -o "$B/broccoli_lib.o" broccoli_lib.cpp
ar rcs "$B/libBROCCOLI_LIB.a" "$B/broccoli_lib.o"
cd "$ROOT/code/Bash_Wrapper"
g++ -no-pie RandomiseGroupLevel.cpp -I"$INC_CL" -DCL_TARGET_OPENCL_VERSION=120 -I../BROCCOLI_LIB -I../BROCCOLI_LIB/Eigen -I../BROCCOLI_LIB/clBLASLinux \
  -Inifticlib-2.0.0/niftilib -Inifticlib-2.0.0/znzlib -L"$B" -L../BROCCOLI_LIB/clBLASLinux -Lnifticlib-2.0.0/lib \
  -lBROCCOLI_LIB -lOpenCL -lclBLAS -lniftiio -lznz -lz -O2 -fopenmp -w -o "$B/RandomiseGroupLevel"
echo "Built $B/RandomiseGroupLevel"
echo "Run with: export BROCCOLI_DIR=$ROOT/ LD_LIBRARY_PATH=$ROOT/code/BROCCOLI_LIB/clBLASLinux"
