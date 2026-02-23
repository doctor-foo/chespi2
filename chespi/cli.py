"""
cli.py — CheSPI command-line interface.

Entry point: chespi (defined in pyproject.toml)

Usage examples
--------------
# Full pipeline from BMRB ID
chespi 19482

# From a local NMR-STAR file
chespi path/to/entry.str -o my_output/

# POTENCI only (no CheZOD, no CheSPI)
chespi 19482 --skip-chezod

# CheZOD Z-scores only (no CheSPI SS prediction)
chespi 19482 --skip-chespi

# With structure coloring (write PML script)
chespi 19482 --pdb 2b97.pdb --color-by chezod

# With headless PyMOL rendering
chespi 19482 --pdb 2b97.pdb --render

# TSV output (default), or CSV, or legacy space-separated
chespi 19482 --fmt tsv
chespi 19482 --fmt csv
chespi 19482 --fmt space
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from numpy import array, isnan


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='chespi',
        description='CheSPI: chemical shift secondary structure population inference',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        'input',
        metavar='INPUT',
        help='BMRB entry ID (integer) or path to an NMR-STAR 3.1 file',
    )

    p.add_argument(
        '-o', '--output',
        metavar='DIR',
        default=None,
        help='Output directory (default: ./chespi_<ID>)',
    )

    # ---- step control -------------------------------------------------------
    step = p.add_argument_group('pipeline step control')
    step.add_argument(
        '--skip-chezod',
        action='store_true',
        help='Run POTENCI only; skip CheZOD and CheSPI',
    )
    step.add_argument(
        '--skip-chespi',
        action='store_true',
        help='Run POTENCI + CheZOD; skip CheSPI SS prediction',
    )

    # ---- re-referencing -----------------------------------------------------
    reref = p.add_argument_group('chemical shift re-referencing')
    reref.add_argument(
        '--min-aic',
        type=float,
        default=5.0,
        metavar='FLOAT',
        help='AIC threshold for accepting an offset correction',
    )
    reref.add_argument(
        '--no-reref',
        action='store_true',
        help='Disable re-referencing (equivalent to --min-aic 999)',
    )

    # ---- structure coloring -------------------------------------------------
    struct = p.add_argument_group('structure coloring (optional)')
    struct.add_argument(
        '--pdb',
        metavar='FILE',
        default=None,
        help='PDB file to colour by CheSPI values',
    )
    struct.add_argument(
        '--color-by',
        choices=['chezod', 'pc1', 'pc2', 'pc3'],
        default='chezod',
        help='Property to use for structure colouring',
    )
    struct.add_argument(
        '--render',
        action='store_true',
        help='Render a PNG via pymol-open-source (must be installed)',
    )

    # ---- plots --------------------------------------------------------------
    plots = p.add_argument_group('plots')
    plots.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip matplotlib PDF output',
    )
    plots.add_argument(
        '--plot-2d',
        action='store_true',
        help='Add a 2D PC1 vs PC2 scatter panel to the plot',
    )

    # ---- output format ------------------------------------------------------
    fmt = p.add_argument_group('output format')
    fmt.add_argument(
        '--fmt',
        choices=['tsv', 'csv', 'space'],
        default='tsv',
        help='Output file format for tabular data',
    )

    # ---- misc ---------------------------------------------------------------
    p.add_argument('-v', '--verbose', action='store_true', help='Verbose output')

    return p


def _get_entry_id(input_str: str) -> str:
    s = str(input_str).strip()
    if s.isdigit():
        return s
    return Path(s).stem


def run(args) -> None:
    """Execute the CheSPI pipeline according to parsed arguments."""
    from chespi import io, potenci, chezod, visualization

    # Resolve output directory
    entry_id = _get_entry_id(args.input)
    outdir = Path(args.output) if args.output else Path(f'chespi_{entry_id}')
    outdir.mkdir(parents=True, exist_ok=True)
    print(f'Output directory: {outdir}')

    # Adjust min_aic if --no-reref
    min_aic = 999.0 if args.no_reref else args.min_aic

    # ------------------------------------------------------------------
    # Step 1: Load entry + compute POTENCI random-coil predictions
    # ------------------------------------------------------------------
    print(f'\n[1/3] Loading entry: {args.input}')
    entry = io.load_entry(args.input)
    seq, first_resnum = io.get_sequence(entry)
    conditions = io.get_sample_conditions(entry)

    if args.verbose:
        print(f'  Sequence length : {len(seq)}')
        print(f'  First residue # : {first_resnum}')
        print(f'  Conditions      : {conditions}')

    print(f'  Running POTENCI (pH={conditions["pH"]}, '
          f'T={conditions["temperature"]} K, '
          f'I={conditions["ion"]} M)')
    pred_shifts = potenci.predict(seq, **conditions)

    if args.skip_chezod:
        print('  --skip-chezod: stopping after POTENCI.')
        return

    # ------------------------------------------------------------------
    # Step 2: CheZOD Z-scores + CheSPI PCs
    # ------------------------------------------------------------------
    print(f'\n[2/3] Computing CheZOD Z-scores and CheSPI components')
    obs_shifts = io.get_chemical_shifts(entry)

    if args.verbose:
        total = sum(obs_shifts.counts.values())
        print(f'  Total backbone shifts (H/C/N): {total}')

    result = chezod.compute(obs_shifts, pred_shifts, seq, min_aic=min_aic)

    # Write CheZOD outputs
    io.write_zscores(outdir, entry_id,
                     result.seq, result.residues,
                     result.zscores, result.pc1, result.pc2, result.pc3,
                     result.mini, fmt=args.fmt)
    io.write_shifts(outdir, entry_id, result.shift_records, fmt=args.fmt)

    # Compute colours (needed for colors.txt and PML script)
    pc1s = array(result.pc1, dtype=float)
    pc2s = array(result.pc2, dtype=float)
    colors = visualization.getseccol(pc1s, pc2s)   # (r_arr, g_arr, b_arr)
    # per-residue format expected by io functions
    colors_per_res = list(zip(colors[0], colors[1], colors[2]))
    io.write_colors(outdir, entry_id, result.residues, colors_per_res, fmt=args.fmt)

    # 4-state populations from PC space
    probs_4 = visualization.getprobs(pc1s, pc2s)
    io.write_populations(outdir, entry_id, result.seq, result.residues,
                         probs_4, fmt=args.fmt)

    # PML script (always written alongside colors.txt)
    io.write_pml_script(outdir, entry_id, result.residues, colors_per_res)

    if args.skip_chespi:
        print('  --skip-chespi: stopping after CheZOD.')
        if not args.no_plot:
            visualization.plot(outdir, entry_id, result,
                               ss_result=None, plot_2d=args.plot_2d)
        if args.pdb:
            _color_structure(args, result, None, colors, outdir, entry_id,
                             visualization)
        return

    # ------------------------------------------------------------------
    # Step 3: CheSPI 8-state / 3-state SS prediction
    # ------------------------------------------------------------------
    from chespi import prediction

    print(f'\n[3/3] Running CheSPI genetic algorithm for SS prediction')
    ss_result = prediction.predict(result, first_resnum=first_resnum)

    # Write all SS outputs
    seq_str = ''.join(ss_result.seq)
    io.write_probs8(outdir, entry_id, ss_result.seq, ss_result.resi_list,
                    ss_result.probs8, fmt=args.fmt)
    io.write_probs3(outdir, entry_id, ss_result.seq, ss_result.resi_list,
                    ss_result.probs3, fmt=args.fmt)
    io.write_max8(outdir, entry_id, ss_result.seq, ss_result.resi_list,
                  ss_result.s8_list, ss_result.prob8_list, ss_result.post_list,
                  fmt=args.fmt)
    io.write_max3(outdir, entry_id, ss_result.seq, ss_result.resi_list,
                  ss_result.s3_list, ss_result.prob3_list, ss_result.post_list,
                  fmt=args.fmt)
    io.write_summary8(outdir, entry_id, seq_str, ss_result.conf8,
                      ''.join(ss_result.s8_list))
    io.write_summary3(outdir, entry_id, seq_str, ss_result.conf3,
                      ''.join(ss_result.s3_list))

    print(f'  Best SS8: {"".join(ss_result.s8_list)}')

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    if not args.no_plot:
        visualization.plot(outdir, entry_id, result,
                           ss_result=ss_result, plot_2d=args.plot_2d)

    # ------------------------------------------------------------------
    # Structure coloring (optional)
    # ------------------------------------------------------------------
    if args.pdb:
        _color_structure(args, result, ss_result, colors, outdir, entry_id,
                         visualization)

    print(f'\nDone.  Outputs in {outdir}/')


def _color_structure(args, result, ss_result, colors, outdir, entry_id,
                     visualization) -> None:
    """Compute per-residue colour array and write PML / optionally render."""
    import numpy as np

    resi   = result.residues
    pc1s   = np.array(result.pc1, dtype=float)
    pc2s   = np.array(result.pc2, dtype=float)

    if args.color_by == 'chezod':
        # Map Z-score to blue (low) → red (high) via PC proxy
        col = colors
    elif args.color_by == 'pc1':
        # Map pc1 range → [0,1] for all 3 channels independently
        v = np.clip(pc1s, -12, 12)
        v_n = (v + 12) / 24.0
        col = (v_n, np.ones_like(v_n) * 0.5, 1.0 - v_n)
    elif args.color_by == 'pc2':
        v = np.clip(pc2s, -8, 8)
        v_n = (v + 8) / 16.0
        col = (v_n, 1.0 - v_n, np.ones_like(v_n) * 0.5)
    elif args.color_by == 'pc3':
        pc3s = np.array(result.pc3, dtype=float)
        v = np.clip(pc3s, -8, 8)
        v_n = (v + 8) / 16.0
        col = (np.ones_like(v_n) * 0.5, v_n, 1.0 - v_n)
    else:
        col = colors

    visualization.color_structure(args.pdb, resi, col, outdir, entry_id,
                                  render=args.render)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as exc:
        print(f'Error: {exc}', file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
