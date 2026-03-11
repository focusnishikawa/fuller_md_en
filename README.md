# fuller_md — Fullerene Crystal NPT Molecular Dynamics

NPT molecular dynamics simulation codes for C60/C70/C72/C74/C76/C84 fullerene crystals.
Three force field models (LJ rigid-body, molecular mechanics, AIREBO) with
Serial / OpenMP / OpenACC GPU modes selectable at compile time.

## Other Versions

| Repository | Language | Description |
|-----------|----------|-------------|
| [fuller_md](https://github.com/focusnishikawa/fuller_md) | C++ (Japanese) | C++ version, Japanese |
| [fuller_md_en](https://github.com/focusnishikawa/fuller_md_en) | C++ (English) | This repository |
| [fuller_md_Julia](https://github.com/focusnishikawa/fuller_md_Julia) | Julia (English) | Julia/JACC.jl version, English |
| [fuller_md_Julia_ja](https://github.com/focusnishikawa/fuller_md_Julia_ja) | Julia (Japanese) | Julia/JACC.jl version, Japanese |

## Directory Structure

```
fuller_md/
├── README.md              ← This file
├── FullereneLib/          ← Fullerene molecular coordinate data (.cc1)
│   ├── C60-76/            ←   C60(Ih), C70(D5h), C72(D6d), C74(D3h), C76(D2,Td)
│   └── C84/               ←   C84 No.01–No.24 (24 isomers)
├── src/                   ← Source code and scripts
│   ├── Build_fuller.sh    ←   Build script
│   ├── Test_fuller.sh     ←   Validation test script
│   ├── fuller_LJ_npt_md_core_serial.cpp          [1] LJ rigid-body core (Serial only)
│   ├── fuller_LJ_npt_md_core_serial_omp_acc.cpp   [2] LJ rigid-body core (Serial/OMP/ACC)
│   ├── fuller_LJ_npt_md_serial_omp_acc.cpp        [3] LJ rigid-body full (Serial/OMP/ACC)
│   ├── fuller_LJ_npt_mmmd_serial_omp_acc.cpp      [4] Molecular mechanics full (Serial/OMP/ACC)
│   └── fuller_airebo_npt_md_serial_omp_acc.cpp    [5] AIREBO full (Serial/OMP/ACC)
└── bin/                   ← Executable output directory (auto-created at build)
```

## Source Files

### Core Versions [1][2] — For Learning and Benchmarking

Parameters are fixed in source code (T=300K, P=0GPa, dt=1fs, 1000 steps).
Only `nc` (cell size) as argument. No restart or OVITO output.

| # | File | Description | Parallelization |
|---|------|-------------|-----------------|
| 1 | `fuller_LJ_npt_md_core_serial.cpp` | LJ rigid-body Serial only. Includes parallelization guide comments | Serial |
| 2 | `fuller_LJ_npt_md_core_serial_omp_acc.cpp` | LJ rigid-body 3-mode unified. `#ifdef` compile-time switching | Serial/OMP/ACC |

### Full Versions [3][4][5] — For Production Calculations

All runtime options supported. Restart save/load, OVITO XYZ output, stop control (abort.md/stop.md) implemented.

| # | File | Force Field | Default dt |
|---|------|-------------|------------|
| 3 | `fuller_LJ_npt_md_serial_omp_acc.cpp` | LJ rigid-body intermolecular potential | 1.0 fs |
| 4 | `fuller_LJ_npt_mmmd_serial_omp_acc.cpp` | Molecular mechanics (Bond+Angle+Dihedral+Improper+LJ+Coulomb) | 0.1 fs |
| 5 | `fuller_airebo_npt_md_serial_omp_acc.cpp` | AIREBO (REBO-II + LJ) | 0.5 fs |

## Build Instructions

### Prerequisites

- **Serial/OpenMP**: GCC 12+ (g++) or equivalent C++17 compiler
- **OpenACC GPU**: NVIDIA HPC SDK (nvc++)
- macOS: Homebrew GCC (`brew install gcc`) required (clang++ does not support `-fopenmp`)

### Build Script

```bash
cd fuller_md

# Build Serial + OpenMP (default)
src/Build_fuller.sh

# Serial only
src/Build_fuller.sh serial

# OpenMP only
src/Build_fuller.sh omp

# OpenACC GPU only
src/Build_fuller.sh acc

# All modes (Serial + OpenMP + OpenACC)
src/Build_fuller.sh all

# Clean build artifacts
src/Build_fuller.sh clean
```

The script auto-detects Homebrew GCC (`g++-15`, `g++-14`, ...) on macOS.
You can also specify the compiler explicitly via the `CXX` environment variable:

```bash
CXX=/opt/gcc/bin/g++ src/Build_fuller.sh
```

### Generated Executables

| Executable | Source | Mode |
|-----------|--------|------|
| `fuller_LJ_core_serial_pure` | [1] | Serial |
| `fuller_LJ_core_serial` | [2] | Serial |
| `fuller_LJ_core_omp` | [2] | OpenMP |
| `fuller_LJ_core_gpu` | [2] | OpenACC |
| `fuller_LJ_npt_md_serial` | [3] | Serial |
| `fuller_LJ_npt_md_omp` | [3] | OpenMP |
| `fuller_LJ_npt_md_gpu` | [3] | OpenACC |
| `fuller_LJ_npt_mmmd_serial` | [4] | Serial |
| `fuller_LJ_npt_mmmd_omp` | [4] | OpenMP |
| `fuller_LJ_npt_mmmd_gpu` | [4] | OpenACC |
| `fuller_airebo_npt_md_serial` | [5] | Serial |
| `fuller_airebo_npt_md_omp` | [5] | OpenMP |
| `fuller_airebo_npt_md_gpu` | [5] | OpenACC |

## Validation Tests

```bash
cd fuller_md
src/Test_fuller.sh
```

Runs all executables with minimal steps and verifies basic operation including OVITO output and restart save.
GPU versions are automatically skipped if not built.

## Usage

### Core Versions

```bash
cd fuller_md

# Default (3x3x3, N=108 molecules, 1000 steps)
bin/fuller_LJ_core_serial_pure

# Specify cell size (5x5x5, N=500 molecules)
bin/fuller_LJ_core_omp 5
bin/fuller_LJ_core_omp --cell=5
```

### Full Versions — Basic Execution

```bash
cd fuller_md

# LJ rigid-body (default: C60, FCC 3x3x3, 298K, 0GPa, 10000 steps)
bin/fuller_LJ_npt_md_serial

# Specify temperature, pressure, and steps
bin/fuller_LJ_npt_md_omp --temp=500 --pres=1.0 --step=50000

# Cold start (4K) + warmup + production
bin/fuller_LJ_npt_md_serial --coldstart=2000 --warmup=3000 --step=20000
```

### Full Versions — OVITO Output

Output XYZ trajectory files at specified intervals for visualization with OVITO.

```bash
# Output every 100 steps
bin/fuller_LJ_npt_md_omp --step=10000 --ovito=100
# → ovito_traj_LJ_omp_*.xyz is generated

# Molecular mechanics version
bin/fuller_LJ_npt_mmmd_omp --step=20000 --ovito=200

# AIREBO version
bin/fuller_airebo_npt_md_omp --step=10000 --ovito=100
```

### Full Versions — Restart

Save restart files at specified intervals. Also auto-saved at the final step.

```bash
# Save restart every 5000 steps
bin/fuller_LJ_npt_md_serial --step=50000 --restart=5000
# → restart_LJ_serial_*_00005000.rst, ..._00010000.rst, ... are generated

# Resume from restart file
bin/fuller_LJ_npt_md_serial --resfile=restart_LJ_serial_00025000.rst

# OVITO + restart combined
bin/fuller_LJ_npt_md_omp --step=100000 --ovito=500 --restart=10000
```

### Full Versions — Stop Control

Create a directory in the current working directory during execution to control the simulation:

```bash
# Stop immediately (saves restart if enabled, then exits)
mkdir abort.md

# Stop at the next restart checkpoint
mkdir stop.md
```

### Full Versions — Help

```bash
bin/fuller_LJ_npt_md_serial --help
bin/fuller_LJ_npt_mmmd_serial --help
bin/fuller_airebo_npt_md_serial --help
```

## Runtime Options (Full Versions)

| Option | Description | Default |
|--------|-------------|---------|
| `--help` | Show help | — |
| `--fullerene=<name>` | Fullerene species | C60 |
| `--crystal=<fcc\|hcp\|bcc>` | Crystal structure | fcc |
| `--cell=<nc>` | Unit cell repeats | 3 |
| `--temp=<K>` | Target temperature [K] | 298.0 |
| `--pres=<GPa>` | Target pressure [GPa] | 0.0 |
| `--step=<N>` | Production steps | 10000 |
| `--dt=<fs>` | Time step [fs] | Force-field dependent |
| `--init_scale=<s>` | Lattice constant scale factor | 1.0 |
| `--seed=<n>` | Random seed | 42 |
| `--coldstart=<N>` | Cold-start steps at 4K | 0 |
| `--warmup=<N>` | Warmup ramp steps 4K→T | 0 |
| `--from=<step>` | Averaging start step | Auto (3/4 point) |
| `--to=<step>` | Averaging end step | nsteps |
| `--mon=<N>` | Monitoring output interval | Auto |
| `--warmup_mon=<mode>` | Output frequency during warmup (norm\|freq\|some) | norm |
| `--ovito=<N>` | OVITO XYZ output interval (0=off) | 0 |
| `--ofile=<filename>` | OVITO output filename (LJ version only) | Auto-generated |
| `--restart=<N>` | Restart save interval (0=off) | 0 |
| `--resfile=<path>` | Resume from restart file | — |
| `--libdir=<path>` | Fullerene library directory | FullereneLib |

### Molecular Mechanics [4] Additional Options

| Option | Description | Default |
|--------|-------------|---------|
| `--ff_kb=<kcal/mol>` | Bond stretching force constant | 469.0 |
| `--ff_kth=<kcal/mol>` | Angle bending force constant | 63.0 |
| `--ff_v2=<kcal/mol>` | Dihedral torsion force constant | 14.5 |
| `--ff_kimp=<kcal/mol>` | Improper dihedral force constant | 15.0 |

## Supported Fullerenes

Specified via the `--fullerene=` option. Coordinate data stored in `FullereneLib/`.

### C60–76 Series

| Specifier | Atoms | Symmetry |
|-----------|-------|----------|
| `C60` (default) | 60 | Ih |
| `C70` | 70 | D5h |
| `C72` | 72 | D6d |
| `C74` | 74 | D3h |
| `C76:D2` | 76 | D2 |
| `C76:Td` | 76 | Td |

### C84 Series (24 Isomers)

Specify as `--fullerene=C84:<number>` or `--fullerene=C84:<number>:<symmetry>`.

| Specifier | Symmetry | Notes |
|-----------|----------|-------|
| `C84:1` | D2 | |
| `C84:2` | C2 | |
| `C84:3` | Cs | |
| `C84:4` | D2d | |
| `C84:5` | D2 | |
| `C84:6` | C2v | |
| `C84:7` | C2v | |
| `C84:8` | C2 | |
| `C84:9` | C2 | |
| `C84:10` | Cs | |
| `C84:11` | C2 | |
| `C84:12` | C1 | |
| `C84:13` | C2 | |
| `C84:14` | Cs | |
| `C84:15` | Cs | |
| `C84:16` | Cs | |
| `C84:17` | C2v | |
| `C84:18` | C2v | |
| `C84:19` | D3d | |
| `C84:20` | Td | Stable isomer |
| `C84:21` | D2 | |
| `C84:22` | D2 | Stable isomer |
| `C84:23` | D2d | Stable isomer |
| `C84:24` | D6h | |

## Physical Model

- **Ensemble**: NPT (isothermal-isobaric)
- **Thermostat**: Nose-Hoover chain
- **Barostat**: Parrinello-Rahman
- **Time integration**: Velocity-Verlet
- **Periodic boundary conditions**: 3D (triclinic cell)
- **Neighbor list**: Symmetric full list (no Newton's 3rd law — avoids GPU write conflicts)

### LJ Rigid-Body Versions [1][2][3]

- Intermolecular: Lennard-Jones (C-C, sigma=3.4A, epsilon=2.63meV)
- Rigid-body rotation: Quaternion representation
- Degrees of freedom: Translational(3) + Rotational(3) × N molecules

### Molecular Mechanics Version [4]

- Intramolecular: Bond stretching + Angle bending + Dihedral + Improper
- Intermolecular: LJ + Coulomb (when charges are present)
- All-atom degrees of freedom

### AIREBO Version [5]

- REBO-II (Brenner 2002): Covalent bonds (bond-order potential)
- LJ (Stuart 2000): Intermolecular van der Waals
- All-atom degrees of freedom

## Unit System

| Quantity | Unit |
|----------|------|
| Length | A (Angstrom) |
| Mass | amu (atomic mass unit) |
| Energy | eV (electron volt) |
| Time | fs (femtosecond) |
| Temperature | K (Kelvin) |
| Pressure | GPa (gigapascal) |

## Supported Environments

- macOS (Homebrew GCC) — Serial/OpenMP
- Linux (GCC 12+) — Serial/OpenMP
- Linux (NVIDIA HPC SDK + GPU) — Serial/OpenMP/OpenACC
- Tested on: FOCUS supercomputer (GCC 12.2.0 + NVIDIA HPC SDK 24.5)
