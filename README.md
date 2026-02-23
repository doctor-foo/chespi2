# CheSPI
Chemical shift Secondary structure Population Inference

**Publication:**
Jakob Toudahl Nielsen and Frans A A Mulder,
*CheSPI: chemical shift secondary structure population inference.*
J. Biomol. NMR, 2021 Jul;75(6-7):273-291.
https://doi.org/10.1007/s10858-021-00374-w

## What it does

1. Derives principal components from backbone chemical shifts to infer local structure and dynamics (CheZOD Z-scores + CheSPI PCs).
2. Predicts secondary structure populations (helix / sheet / turn / coil) from chemical shifts.
3. Predicts secondary structure DSSP 8-state classes from chemical shifts using a genetic algorithm.

## Installation

Requires Python ≥ 3.9. Install with [uv](https://github.com/astral-sh/uv) (recommended) or pip:

```bash
# with uv
uv pip install .

# with pip
pip install .
```

Optional PyMOL rendering support:

```bash
uv pip install ".[render]"
```

## Usage

```
chespi INPUT [OPTIONS]
```

`INPUT` is either a **BMRB entry ID** (downloaded automatically) or a path to a local **NMR-STAR 3.1 file**.

### Examples

```bash
# Full pipeline: POTENCI → CheZOD → CheSPI SS prediction
chespi 19482

# From a local NMR-STAR file
chespi path/to/entry.str -o my_output/

# POTENCI random-coil predictions only
chespi 19482 --skip-chezod

# CheZOD Z-scores + CheSPI components only (no SS prediction)
chespi 19482 --skip-chespi

# Write a PyMOL coloring script alongside the other outputs
chespi 19482 --pdb 2b97.pdb --color-by chezod

# Render a PNG via headless PyMOL (requires pymol-open-source)
chespi 19482 --pdb 2b97.pdb --render

# Legacy space-separated output format (backward compatible)
chespi 19482 --fmt space

# Add a 2D PC1 vs PC2 scatter panel to the plot
chespi 19482 --plot-2d
```

### All options

| Flag | Default | Description |
|------|---------|-------------|
| `-o / --output DIR` | `chespi_<ID>/` | Output directory |
| `--skip-chezod` | off | Run POTENCI only |
| `--skip-chespi` | off | Run POTENCI + CheZOD only |
| `--min-aic FLOAT` | `5.0` | AIC threshold for re-referencing |
| `--no-reref` | off | Disable re-referencing entirely |
| `--pdb FILE` | — | PDB file for structure coloring |
| `--color-by` | `chezod` | `chezod` / `pc1` / `pc2` / `pc3` |
| `--render` | off | Ray-trace PNG via PyMOL |
| `--no-plot` | off | Skip PDF plot |
| `--plot-2d` | off | Add 2D PC1 vs PC2 scatter panel |
| `--fmt` | `tsv` | Output format: `tsv` / `csv` / `space` |
| `-v / --verbose` | off | Verbose output |

## Output files

All files are written to the output directory (default `chespi_<ID>/`).

| File | Contents |
|------|----------|
| `shifts.txt` | Secondary chemical shifts (obs − POTENCI) |
| `zscores.txt` | CheZOD Z-scores and CheSPI PC1/PC2/PC3 per residue |
| `colors.txt` | CheSPI RGB colors per residue (hex + integer) |
| `populations.txt` | 4-state populations: helix / turn / coil / strand |
| `probs8.txt` | 8-state DSSP probabilities (H G I E - T S B) |
| `probs3.txt` | 3-state probabilities (H S C) |
| `max8.txt` | Best 8-state prediction with probability and posterior |
| `max3.txt` | Best 3-state prediction with probability and posterior |
| `summary8.txt` | One-liner: sequence / confidence string / 8-state prediction |
| `summary3.txt` | One-liner: sequence / confidence string / 3-state prediction |
| `cheSPIplot.pdf` | Multi-panel matplotlib figure |
| `colCheSPI.pml` | PyMOL script for CheSPI structure coloring |
| `structure_<ID>.png` | Ray-traced PNG (only with `--pdb --render`) |

## Re-referencing

CheSPI automatically detects and corrects systematic chemical shift referencing offsets using an AIC criterion. The default threshold (`--min-aic 5.0`) accepts corrections only when the statistical evidence is strong. Use `--no-reref` to disable this entirely.

## Notes on chemical shift coverage

CheSPI will provide secondary structure predictions even with limited chemical shift assignments. For residues with no assigned shifts the prediction falls back to sequence-based priors and will have lower confidence (reflected in the confidence digit in `summary8.txt` / `summary3.txt`).

## Package structure

```
chespi/
├── cli.py            # command-line entry point
├── io.py             # NMR-STAR parsing (pynmrstar) + output writers
├── potenci.py        # POTENCI random-coil shift prediction
├── chezod.py         # CheZOD Z-scores + CheSPI PCs
├── ga.py             # genetic algorithm base classes
├── prediction.py     # 8-state / 3-state SS prediction (GA)
├── visualization.py  # matplotlib plots + PyMOL coloring
└── data/             # pre-computed tables (CSV / JSON / NPY)
```

The original monolithic scripts are preserved in `legacy/` for reference.

```
legacy/
├── cheSPI4c.py                # original Python 2 / NMR-STAR 2.1 version
└── chespi_for_nmrstar31.py   # Python 3 port with NMR-STAR 3.1 support
```
