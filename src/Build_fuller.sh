#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Takeshi Nishikawa
#===========================================================================
#  Build_fuller.sh — Fullerene crystal NPT-MD build script
#
#  Usage:
#    ./Build_fuller.sh              # Serial + OpenMP full build
#    ./Build_fuller.sh serial       # Serial only
#    ./Build_fuller.sh omp          # OpenMP only
#    ./Build_fuller.sh acc          # OpenACC GPU only
#    ./Build_fuller.sh all          # Serial + OpenMP + OpenACC all
#    ./Build_fuller.sh clean        # Delete executables in bin/
#
#  Directory structure:
#    fuller_md/
#    ├── src/           ← This script and source code
#    ├── bin/           ← Executable output directory
#    └── FullereneLib/  ← Fullerene coordinate data
#
#---------------------------------------------------------------------------
#  Build target source files (5 files):
#
#  [1] fuller_LJ_npt_md_core_serial.cpp         — LJ rigid-body core version (Serial only)
#  [2] fuller_LJ_npt_md_core_serial_omp_acc.cpp — LJ rigid-body core version (Serial/OMP/ACC)
#  [3] fuller_LJ_npt_md_serial_omp_acc.cpp      — LJ rigid-body full version (Serial/OMP/ACC)
#  [4] fuller_LJ_npt_mmmd_serial_omp_acc.cpp    — Molecular mechanics full version (Serial/OMP/ACC)
#  [5] fuller_airebo_npt_md_serial_omp_acc.cpp  — AIREBO full version (Serial/OMP/ACC)
#
#  Core version [1][2]: Fixed parameters, only nc (cell size) as argument
#                       No restart capability, no OVITO output
#  Full version [3][4][5]: All runtime options supported
#                          Restart save/resume, OVITO XYZ output supported
#
#---------------------------------------------------------------------------
#  Execution examples — Core version [1][2]:
#
#    bin/fuller_LJ_core_serial_pure       # 3x3x3 (N=108) default
#    bin/fuller_LJ_core_serial_pure 5     # 5x5x5 (N=500)
#    bin/fuller_LJ_core_omp 4             # OpenMP, 4x4x4 (N=256)
#
#---------------------------------------------------------------------------
#  Execution examples — LJ rigid-body full version [3]:
#
#    # Basic execution (C60 FCC 3x3x3, 298K, 10000 steps)
#    bin/fuller_LJ_npt_md_serial
#
#    # Specify temperature, pressure, and number of steps
#    bin/fuller_LJ_npt_md_omp --temp=500 --pres=1.0 --step=50000
#
#    # Cold start + warm-up + production run
#    bin/fuller_LJ_npt_md_serial --coldstart=2000 --warmup=3000 --step=20000
#
#    # OVITO XYZ output (write trajectory every 100 steps)
#    bin/fuller_LJ_npt_md_omp --step=10000 --ovito=100
#
#    # Restart save (every 5000 steps + final step)
#    bin/fuller_LJ_npt_md_serial --step=50000 --restart=5000
#
#    # Resume from restart file
#    bin/fuller_LJ_npt_md_serial --resfile=restart_LJ_serial_00025000.rst
#
#    # Use OVITO + restart simultaneously
#    bin/fuller_LJ_npt_md_omp --step=100000 --ovito=500 --restart=10000
#
#    # Show all options
#    bin/fuller_LJ_npt_md_serial --help
#
#---------------------------------------------------------------------------
#  Execution examples — Molecular mechanics full version [4]:
#
#    bin/fuller_LJ_npt_mmmd_serial --step=20000
#    bin/fuller_LJ_npt_mmmd_omp --temp=500 --step=100000 --ovito=200
#    bin/fuller_LJ_npt_mmmd_serial --step=100000 --restart=10000
#    bin/fuller_LJ_npt_mmmd_serial --resfile=restart_mmmd_serial_00050000.rst
#    bin/fuller_LJ_npt_mmmd_omp --ff_kb=500 --ff_kth=70 --step=20000
#    bin/fuller_LJ_npt_mmmd_serial --help
#
#---------------------------------------------------------------------------
#  Execution examples — AIREBO full version [5]:
#
#    bin/fuller_airebo_npt_md_serial --step=10000
#    bin/fuller_airebo_npt_md_omp --temp=500 --step=50000 --ovito=100
#    bin/fuller_airebo_npt_md_serial --step=50000 --restart=5000
#    bin/fuller_airebo_npt_md_serial --resfile=restart_airebo_serial_00025000.rst
#    bin/fuller_airebo_npt_md_omp --fullerene=C84:20:Td --cell=4 --step=20000
#    bin/fuller_airebo_npt_md_serial --help
#
#---------------------------------------------------------------------------
#  Stop control (common to full version [3][4][5]):
#    Create the following in the current directory during execution:
#      mkdir abort.md   → Stop immediately (saves and exits when restart is enabled)
#      mkdir stop.md    → Stop at the next restart checkpoint
#
#===========================================================================
set -e

# Set paths relative to the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SRCDIR="$SCRIPT_DIR"
BINDIR="$BASE_DIR/bin"
CXXSTD="-std=c++17"
OPT="-O3"

#--- Compiler detection ----------------------------------------------------
# On macOS, g++ is an alias for clang and does not support -fopenmp,
# so we look for a real GCC installed via Homebrew
find_gxx() {
    if [ -n "$CXX" ]; then
        echo "$CXX"
        return
    fi
    for v in 15 14 13 12; do
        for p in /opt/homebrew/bin /usr/local/bin; do
            if [ -x "$p/g++-$v" ]; then
                echo "$p/g++-$v"
                return
            fi
        done
    done
    echo "g++"
}

GXX=$(find_gxx)
NVCXX="nvc++"

#--- Build mode selection --------------------------------------------------
MODE="${1:-serial_omp}"
case "$MODE" in
    serial)     DO_SERIAL=1; DO_OMP=0; DO_ACC=0 ;;
    omp)        DO_SERIAL=0; DO_OMP=1; DO_ACC=0 ;;
    acc|gpu)    DO_SERIAL=0; DO_OMP=0; DO_ACC=1 ;;
    all)        DO_SERIAL=1; DO_OMP=1; DO_ACC=1 ;;
    serial_omp) DO_SERIAL=1; DO_OMP=1; DO_ACC=0 ;;
    clean)
        echo "Cleaning ${BINDIR}/ ..."
        rm -f "${BINDIR}"/*
        echo "Done."
        exit 0
        ;;
    *)
        echo "Usage: $0 [serial|omp|acc|all|clean]"
        exit 1
        ;;
esac

mkdir -p "$BINDIR"

#--- Compiler verification -------------------------------------------------
echo "================================================================"
echo "  Build_fuller.sh"
echo "================================================================"
echo "  g++ compiler : $GXX"
if [ "$DO_ACC" -eq 1 ]; then
    if ! command -v "$NVCXX" &>/dev/null; then
        echo "  WARNING: $NVCXX not found — OpenACC builds will be skipped"
        DO_ACC=0
    else
        echo "  nvc++ compiler: $NVCXX"
    fi
fi
echo "  Build mode   : serial=$DO_SERIAL omp=$DO_OMP acc=$DO_ACC"
echo "  Source dir   : $SRCDIR/"
echo "  Output dir   : $BINDIR/"
echo "================================================================"
echo ""

OK=0
FAIL=0

#--- Build function --------------------------------------------------------
build() {
    local compiler="$1"
    local flags="$2"
    local src="$3"
    local out="$4"

    printf "  %-50s ... " "$out"
    if $compiler $CXXSTD $OPT $flags -o "${BINDIR}/${out}" "${SRCDIR}/${src}" -lm 2>/tmp/build_fuller_err.txt; then
        echo "OK"
        OK=$((OK + 1))
    else
        echo "FAIL"
        cat /tmp/build_fuller_err.txt | head -5
        FAIL=$((FAIL + 1))
    fi
}

#===========================================================================
#  1) fuller_LJ_npt_md_core_serial.cpp  (Serial only)
#===========================================================================
echo "[1/5] fuller_LJ_npt_md_core_serial.cpp (Serial only)"
if [ "$DO_SERIAL" -eq 1 ]; then
    build "$GXX" "" \
        "fuller_LJ_npt_md_core_serial.cpp" \
        "fuller_LJ_core_serial_pure"
fi
echo ""

#===========================================================================
#  2) fuller_LJ_npt_md_core_serial_omp_acc.cpp  (Serial/OMP/ACC)
#===========================================================================
echo "[2/5] fuller_LJ_npt_md_core_serial_omp_acc.cpp"
if [ "$DO_SERIAL" -eq 1 ]; then
    build "$GXX" "-Wno-unknown-pragmas" \
        "fuller_LJ_npt_md_core_serial_omp_acc.cpp" \
        "fuller_LJ_core_serial"
fi
if [ "$DO_OMP" -eq 1 ]; then
    build "$GXX" "-fopenmp -Wno-unknown-pragmas" \
        "fuller_LJ_npt_md_core_serial_omp_acc.cpp" \
        "fuller_LJ_core_omp"
fi
if [ "$DO_ACC" -eq 1 ]; then
    build "$NVCXX" "-acc -gpu=cc80 -Minfo=accel" \
        "fuller_LJ_npt_md_core_serial_omp_acc.cpp" \
        "fuller_LJ_core_gpu"
fi
echo ""

#===========================================================================
#  3) fuller_LJ_npt_md_serial_omp_acc.cpp  (Serial/OMP/ACC)
#===========================================================================
echo "[3/5] fuller_LJ_npt_md_serial_omp_acc.cpp"
if [ "$DO_SERIAL" -eq 1 ]; then
    build "$GXX" "-Wno-unknown-pragmas" \
        "fuller_LJ_npt_md_serial_omp_acc.cpp" \
        "fuller_LJ_npt_md_serial"
fi
if [ "$DO_OMP" -eq 1 ]; then
    build "$GXX" "-fopenmp -Wno-unknown-pragmas" \
        "fuller_LJ_npt_md_serial_omp_acc.cpp" \
        "fuller_LJ_npt_md_omp"
fi
if [ "$DO_ACC" -eq 1 ]; then
    build "$NVCXX" "-acc -gpu=cc80 -Minfo=accel" \
        "fuller_LJ_npt_md_serial_omp_acc.cpp" \
        "fuller_LJ_npt_md_gpu"
fi
echo ""

#===========================================================================
#  4) fuller_LJ_npt_mmmd_serial_omp_acc.cpp  (Serial/OMP/ACC)
#===========================================================================
echo "[4/5] fuller_LJ_npt_mmmd_serial_omp_acc.cpp"
if [ "$DO_SERIAL" -eq 1 ]; then
    build "$GXX" "-Wno-unknown-pragmas" \
        "fuller_LJ_npt_mmmd_serial_omp_acc.cpp" \
        "fuller_LJ_npt_mmmd_serial"
fi
if [ "$DO_OMP" -eq 1 ]; then
    build "$GXX" "-fopenmp -Wno-unknown-pragmas" \
        "fuller_LJ_npt_mmmd_serial_omp_acc.cpp" \
        "fuller_LJ_npt_mmmd_omp"
fi
if [ "$DO_ACC" -eq 1 ]; then
    build "$NVCXX" "-acc -gpu=cc80 -Minfo=accel" \
        "fuller_LJ_npt_mmmd_serial_omp_acc.cpp" \
        "fuller_LJ_npt_mmmd_gpu"
fi
echo ""

#===========================================================================
#  5) fuller_airebo_npt_md_serial_omp_acc.cpp  (Serial/OMP/ACC)
#===========================================================================
echo "[5/5] fuller_airebo_npt_md_serial_omp_acc.cpp"
if [ "$DO_SERIAL" -eq 1 ]; then
    build "$GXX" "-Wno-unknown-pragmas" \
        "fuller_airebo_npt_md_serial_omp_acc.cpp" \
        "fuller_airebo_npt_md_serial"
fi
if [ "$DO_OMP" -eq 1 ]; then
    build "$GXX" "-fopenmp -Wno-unknown-pragmas" \
        "fuller_airebo_npt_md_serial_omp_acc.cpp" \
        "fuller_airebo_npt_md_omp"
fi
if [ "$DO_ACC" -eq 1 ]; then
    build "$NVCXX" "-acc -gpu=cc80 -Minfo=accel" \
        "fuller_airebo_npt_md_serial_omp_acc.cpp" \
        "fuller_airebo_npt_md_gpu"
fi
echo ""

#--- Results summary -------------------------------------------------------
echo "================================================================"
echo "  Build complete:  OK=$OK  FAIL=$FAIL"
if [ "$OK" -gt 0 ]; then
    echo ""
    echo "  Executables in ${BINDIR}/:"
    ls -lh "${BINDIR}"/ 2>/dev/null | grep -v "^total" | awk '{printf "    %-40s %s\n", $NF, $5}'
fi
echo "================================================================"
echo ""
echo "Execution examples:"
echo ""
echo "  [Core version] Fixed parameters, only nc (cell size) as argument"
echo "    ${BINDIR}/fuller_LJ_core_serial_pure         # Serial, default 3x3x3"
echo "    ${BINDIR}/fuller_LJ_core_serial_pure 5       # Serial, 5x5x5 (N=500)"
echo "    ${BINDIR}/fuller_LJ_core_omp 4               # OpenMP, 4x4x4 (N=256)"
echo ""
echo "  [LJ rigid-body full version] All runtime options supported"
echo "    ${BINDIR}/fuller_LJ_npt_md_serial --help                          # Show help"
echo "    ${BINDIR}/fuller_LJ_npt_md_omp --temp=500 --step=50000            # Specify temperature"
echo "    ${BINDIR}/fuller_LJ_npt_md_serial --step=10000 --ovito=100        # OVITO output"
echo "    ${BINDIR}/fuller_LJ_npt_md_serial --step=50000 --restart=5000     # Restart save"
echo "    ${BINDIR}/fuller_LJ_npt_md_serial --resfile=restart_*.rst         # Restart resume"
echo ""
echo "  [Molecular mechanics full version]"
echo "    ${BINDIR}/fuller_LJ_npt_mmmd_serial --help                        # Show help"
echo "    ${BINDIR}/fuller_LJ_npt_mmmd_omp --step=100000 --ovito=200        # OVITO output"
echo "    ${BINDIR}/fuller_LJ_npt_mmmd_serial --step=100000 --restart=10000 # Restart save"
echo "    ${BINDIR}/fuller_LJ_npt_mmmd_serial --resfile=restart_*.rst       # Restart resume"
echo ""
echo "  [AIREBO full version]"
echo "    ${BINDIR}/fuller_airebo_npt_md_serial --help                      # Show help"
echo "    ${BINDIR}/fuller_airebo_npt_md_omp --step=50000 --ovito=100       # OVITO output"
echo "    ${BINDIR}/fuller_airebo_npt_md_serial --step=50000 --restart=5000 # Restart save"
echo "    ${BINDIR}/fuller_airebo_npt_md_serial --resfile=restart_*.rst     # Restart resume"
echo ""
echo "  [Stop control] Common to full version [3][4][5]"
echo "    mkdir abort.md   # Stop immediately (saves and exits when restart is enabled)"
echo "    mkdir stop.md    # Stop at the next restart checkpoint"
echo ""

exit $FAIL
