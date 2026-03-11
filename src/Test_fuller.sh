#!/bin/bash
#===========================================================================
#  Test_fuller.sh — Fullerene crystal NPT-MD validation test script
#
#  Usage:
#    cd fuller_md && src/Test_fuller.sh
#
#  Overview:
#    Runs all execution modules with minimal step counts
#    to verify that they complete successfully.
#    Must be run from the fuller_md/ directory
#    (because relative paths to FullereneLib/ are used).
#
#---------------------------------------------------------------------------
#  Available fullerenes (--fullerene= option):
#
#  * C60-76 series (FullereneLib/C60-76/):
#    --fullerene=C60       C60 Buckyball (Ih symmetry, 60 atoms) *Default
#    --fullerene=C70       C70 (D5h symmetry, 70 atoms)
#    --fullerene=C72       C72 (D6d symmetry, 72 atoms)
#    --fullerene=C74       C74 (D3h symmetry, 74 atoms)
#    --fullerene=C76:D2    C76 isomer (D2 symmetry, 76 atoms)
#    --fullerene=C76:Td    C76 isomer (Td symmetry, 76 atoms)
#
#  * C84 series (FullereneLib/C84/, 24 isomers):
#    --fullerene=C84:1     C84 No.01 (D2 symmetry)
#    --fullerene=C84:2     C84 No.02 (C2 symmetry)
#    --fullerene=C84:3     C84 No.03 (Cs symmetry)
#    --fullerene=C84:4     C84 No.04 (D2d symmetry)
#    --fullerene=C84:5     C84 No.05 (D2 symmetry)
#    --fullerene=C84:6     C84 No.06 (C2v symmetry)
#    --fullerene=C84:7     C84 No.07 (C2v symmetry)
#    --fullerene=C84:8     C84 No.08 (C2 symmetry)
#    --fullerene=C84:9     C84 No.09 (C2 symmetry)
#    --fullerene=C84:10    C84 No.10 (Cs symmetry)
#    --fullerene=C84:11    C84 No.11 (C2 symmetry)
#    --fullerene=C84:12    C84 No.12 (C1 symmetry)
#    --fullerene=C84:13    C84 No.13 (C2 symmetry)
#    --fullerene=C84:14    C84 No.14 (Cs symmetry)
#    --fullerene=C84:15    C84 No.15 (Cs symmetry)
#    --fullerene=C84:16    C84 No.16 (Cs symmetry)
#    --fullerene=C84:17    C84 No.17 (C2v symmetry)
#    --fullerene=C84:18    C84 No.18 (C2v symmetry)
#    --fullerene=C84:19    C84 No.19 (D3d symmetry)
#    --fullerene=C84:20    C84 No.20 (Td symmetry)   *One of the most stable isomers
#    --fullerene=C84:21    C84 No.21 (D2 symmetry)
#    --fullerene=C84:22    C84 No.22 (D2 symmetry)   *One of the most stable isomers
#    --fullerene=C84:23    C84 No.23 (D2d symmetry)  *One of the most stable isomers
#    --fullerene=C84:24    C84 No.24 (D6h symmetry)
#
#    Format with explicit symmetry group: --fullerene=C84:20:Td
#
#---------------------------------------------------------------------------
#  Complete options list (common to full versions [3][4][5]):
#
#    --help                  Show help
#    --fullerene=<name>      Fullerene species (default: C60)
#    --crystal=<fcc|hcp|bcc> Crystal structure (default: fcc)
#    --cell=<nc>             Unit cell repetition count (default: 3)
#    --temp=<K>              Target temperature [K] (default: 298.0)
#    --pres=<GPa>            Target pressure [GPa] (default: 0.0)
#    --step=<N>              Production run steps (default: 10000)
#    --dt=<fs>               Time step [fs] (LJ: 1.0, MMMD: 0.1, AIREBO: 0.5)
#    --init_scale=<s>        Lattice constant scale factor (default: 1.0)
#    --seed=<n>              Random seed (default: 42)
#    --coldstart=<N>         Low-temperature (4K) steps (default: 0)
#    --warmup=<N>            Heating steps 4K->T (default: 0)
#    --from=<step>           Averaging start step (default: 3/4 of production run)
#    --to=<step>             Averaging end step (default: nsteps)
#    --mon=<N>               Monitoring output interval (default: auto)
#    --warmup_mon=<mode>     Output frequency during warmup norm|freq|some (default: norm)
#    --ovito=<N>             OVITO XYZ output interval (0=disabled, default: 0)
#    --ofile=<filename>      OVITO output filename (LJ version only, default: auto)
#    --restart=<N>           Restart save interval (0=disabled, default: 0)
#    --resfile=<path>        Resume from restart file
#    --libdir=<path>         Fullerene library (default: FullereneLib)
#
#  Molecular mechanics version [4] additional options:
#    --ff_kb=<kcal/mol>      Bond stretching force constant (default: 469.0)
#    --ff_kth=<kcal/mol>     Angle bending force constant (default: 63.0)
#    --ff_v2=<kcal/mol>      Dihedral angle force constant (default: 14.5)
#    --ff_kimp=<kcal/mol>    Improper dihedral force constant (default: 15.0)
#
#---------------------------------------------------------------------------
#  Comprehensive execution examples:
#
#  * LJ rigid-body version (dt=1.0fs):
#    bin/fuller_LJ_npt_md_omp \
#      --fullerene=C60 --crystal=fcc --cell=3 \
#      --temp=298 --pres=0.0 --step=50000 --dt=1.0 \
#      --init_scale=1.0 --seed=42 \
#      --coldstart=2000 --warmup=3000 \
#      --from=40000 --to=50000 --mon=500 --warmup_mon=norm \
#      --ovito=100 --ofile=my_traj.xyz \
#      --restart=5000 --libdir=FullereneLib
#
#  * LJ rigid-body version restart resume:
#    bin/fuller_LJ_npt_md_omp --resfile=restart_LJ_omp_00025000.rst
#
#  * Molecular mechanics version (dt=0.1fs):
#    bin/fuller_LJ_npt_mmmd_omp \
#      --fullerene=C70 --crystal=fcc --cell=3 \
#      --temp=500 --pres=1.0 --step=100000 --dt=0.1 \
#      --init_scale=1.0 --seed=123 \
#      --coldstart=5000 --warmup=5000 \
#      --from=80000 --to=100000 --mon=1000 --warmup_mon=freq \
#      --ovito=200 --restart=10000 --libdir=FullereneLib \
#      --ff_kb=469 --ff_kth=63 --ff_v2=14.5 --ff_kimp=15.0
#
#  * Molecular mechanics version restart resume:
#    bin/fuller_LJ_npt_mmmd_omp --resfile=restart_mmmd_omp_00050000.rst
#
#  * AIREBO version (dt=0.5fs):
#    bin/fuller_airebo_npt_md_omp \
#      --fullerene=C84:20:Td --crystal=fcc --cell=3 \
#      --temp=300 --pres=0.0 --step=50000 --dt=0.5 \
#      --init_scale=1.0 --seed=99 \
#      --coldstart=3000 --warmup=2000 \
#      --from=40000 --to=50000 --mon=500 --warmup_mon=some \
#      --ovito=100 --restart=5000 --libdir=FullereneLib
#
#  * AIREBO version restart resume:
#    bin/fuller_airebo_npt_md_omp --resfile=restart_airebo_omp_00025000.rst
#
#  * Examples with different fullerene species:
#    bin/fuller_LJ_npt_md_omp --fullerene=C70 --step=10000
#    bin/fuller_LJ_npt_md_omp --fullerene=C72 --step=10000
#    bin/fuller_LJ_npt_md_omp --fullerene=C74 --step=10000
#    bin/fuller_LJ_npt_md_omp --fullerene=C76:D2 --step=10000
#    bin/fuller_LJ_npt_md_omp --fullerene=C76:Td --step=10000
#    bin/fuller_LJ_npt_md_omp --fullerene=C84:20:Td --step=10000
#    bin/fuller_LJ_npt_md_omp --fullerene=C84:23:D2d --step=10000
#    bin/fuller_LJ_npt_md_omp --fullerene=C84:22:D2 --step=10000
#
#  * Stop control:
#    mkdir abort.md   # Stop immediately (saves and exits when restart is enabled)
#    mkdir stop.md    # Stop at next restart checkpoint
#
#===========================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BINDIR="$BASE_DIR/bin"

# Tests run from the fuller_md/ directory (for FullereneLib reference)
cd "$BASE_DIR"

echo "================================================================"
echo "  Test_fuller.sh — Validation test"
echo "================================================================"
echo "  Working dir : $(pwd)"
echo "  Bin dir     : ${BINDIR}/"
echo "================================================================"
echo ""

PASS=0
FAIL=0
SKIP=0

#--- Test function ------------------------------------------------------------
run_test() {
    local name="$1"
    local exe="$2"
    shift 2
    local args=("$@")

    printf "  %-52s ... " "$name"
    if [ ! -x "$exe" ]; then
        echo "SKIP (not built)"
        SKIP=$((SKIP + 1))
        return
    fi
    if "$exe" "${args[@]}" > /tmp/test_fuller_out.txt 2>&1; then
        # Check if output contains "Done" (indicator of successful completion)
        if grep -q "Done" /tmp/test_fuller_out.txt; then
            echo "PASS"
            PASS=$((PASS + 1))
        else
            echo "FAIL (no 'Done' in output)"
            tail -3 /tmp/test_fuller_out.txt
            FAIL=$((FAIL + 1))
        fi
    else
        echo "FAIL (exit code $?)"
        tail -3 /tmp/test_fuller_out.txt
        FAIL=$((FAIL + 1))
    fi
}

#===========================================================================
#  [1] fuller_LJ_core_serial_pure — Core version, serial only
#      Arguments: [nc] only. No restart, no OVITO.
#===========================================================================
echo "[1/5] LJ Core Serial Pure (100 steps)"
run_test "fuller_LJ_core_serial_pure" \
    "$BINDIR/fuller_LJ_core_serial_pure"
echo ""

#===========================================================================
#  [2] fuller_LJ_core — Core version Serial/OMP/ACC
#      Arguments: [nc] only. No restart, no OVITO.
#===========================================================================
echo "[2/5] LJ Core Serial/OMP (100 steps)"
run_test "fuller_LJ_core_serial" \
    "$BINDIR/fuller_LJ_core_serial"
run_test "fuller_LJ_core_omp" \
    "$BINDIR/fuller_LJ_core_omp"
run_test "fuller_LJ_core_gpu" \
    "$BINDIR/fuller_LJ_core_gpu"
echo ""

#===========================================================================
#  [3] fuller_LJ_npt_md — LJ rigid-body full version
#      Minimal steps (step=200) for basic verification including restart and OVITO.
#===========================================================================
echo "[3/5] LJ Rigid-body Full (200 steps, with restart+ovito)"
run_test "fuller_LJ_npt_md_serial" \
    "$BINDIR/fuller_LJ_npt_md_serial" \
    --step=200 --mon=100
run_test "fuller_LJ_npt_md_omp" \
    "$BINDIR/fuller_LJ_npt_md_omp" \
    --step=200 --mon=100
run_test "fuller_LJ_npt_md_gpu" \
    "$BINDIR/fuller_LJ_npt_md_gpu" \
    --step=200 --mon=100
run_test "fuller_LJ_npt_md_serial (ovito)" \
    "$BINDIR/fuller_LJ_npt_md_serial" \
    --step=200 --mon=100 --ovito=100
run_test "fuller_LJ_npt_md_serial (restart)" \
    "$BINDIR/fuller_LJ_npt_md_serial" \
    --step=200 --mon=100 --restart=100
echo ""

#===========================================================================
#  [4] fuller_LJ_npt_mmmd — Molecular mechanics full version
#      Minimal steps (step=200) for basic verification including restart and OVITO.
#===========================================================================
echo "[4/5] Molecular Mechanics Full (200 steps, with restart+ovito)"
run_test "fuller_LJ_npt_mmmd_serial" \
    "$BINDIR/fuller_LJ_npt_mmmd_serial" \
    --step=200 --mon=100
run_test "fuller_LJ_npt_mmmd_omp" \
    "$BINDIR/fuller_LJ_npt_mmmd_omp" \
    --step=200 --mon=100
run_test "fuller_LJ_npt_mmmd_gpu" \
    "$BINDIR/fuller_LJ_npt_mmmd_gpu" \
    --step=200 --mon=100
run_test "fuller_LJ_npt_mmmd_serial (ovito)" \
    "$BINDIR/fuller_LJ_npt_mmmd_serial" \
    --step=200 --mon=100 --ovito=100
run_test "fuller_LJ_npt_mmmd_serial (restart)" \
    "$BINDIR/fuller_LJ_npt_mmmd_serial" \
    --step=200 --mon=100 --restart=100
echo ""

#===========================================================================
#  [5] fuller_airebo_npt_md — AIREBO full version
#      Minimal steps (step=200) for basic verification including restart and OVITO.
#===========================================================================
echo "[5/5] AIREBO Full (200 steps, with restart+ovito)"
run_test "fuller_airebo_npt_md_serial" \
    "$BINDIR/fuller_airebo_npt_md_serial" \
    --step=200 --mon=100
run_test "fuller_airebo_npt_md_omp" \
    "$BINDIR/fuller_airebo_npt_md_omp" \
    --step=200 --mon=100
run_test "fuller_airebo_npt_md_gpu" \
    "$BINDIR/fuller_airebo_npt_md_gpu" \
    --step=200 --mon=100
run_test "fuller_airebo_npt_md_serial (ovito)" \
    "$BINDIR/fuller_airebo_npt_md_serial" \
    --step=200 --mon=100 --ovito=100
run_test "fuller_airebo_npt_md_serial (restart)" \
    "$BINDIR/fuller_airebo_npt_md_serial" \
    --step=200 --mon=100 --restart=100
echo ""

#--- Clean up test-generated files --------------------------------------------
echo "Cleaning up test output files..."
rm -f ovito_traj_*.xyz restart_*.rst
echo ""

#--- Results summary ----------------------------------------------------------
echo "================================================================"
echo "  Test complete:  PASS=$PASS  FAIL=$FAIL  SKIP=$SKIP"
echo "================================================================"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
