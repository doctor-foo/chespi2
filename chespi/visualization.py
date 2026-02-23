"""
visualization.py — Matplotlib plots and PyMOL structure coloring.

Public functions
----------------
getseccol(pc1, pc2)
    Map PC1/PC2 values to RGB color tuples.

getprobs(pc1, pc2)
    4-state (E/H/T/N) probability from PC space via spline interpolation.

plot(outdir, entry_id, result, ss_result=None, plot_2d=False)
    Render the main CheSPI figure (1–3 panels) and save as PDF.

color_structure(pdb_path, resi, colors, outdir, entry_id, render=False)
    Write a PyMOL coloring script and optionally ray-trace a PNG.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from numpy import array, clip, linspace, pi, cumsum, zeros


# ---------------------------------------------------------------------------
# Color mapping
# ---------------------------------------------------------------------------

def getseccol(pc1, pc2):
    """Map PC1, PC2 values to RGB tuple(s) for CheSPI coloring.

    Parameters
    ----------
    pc1, pc2 : float or array_like

    Returns
    -------
    tuple (r, g, b) where each component is a float in [0, 1],
    or arrays when pc1/pc2 are arrays.
    """
    pc1 = np.asarray(pc1, dtype=float)
    pc2 = np.asarray(pc2, dtype=float)
    i = (clip(pc1, -12, 12) + 12) / 24.0   # 0 → 1
    j = (clip(pc2, -8,  8)  +  8) / 16.0   # 0 → 1
    return (i, 1 - j, 1 - i)


# ---------------------------------------------------------------------------
# 4-state probability from PC space
# ---------------------------------------------------------------------------

# Loaded lazily
_HISTPROBS: Optional[np.ndarray] = None
_SPLINES: Optional[list] = None


def _load_splines():
    global _HISTPROBS, _SPLINES
    if _SPLINES is not None:
        return _SPLINES
    from scipy.interpolate import RectBivariateSpline
    data_dir = Path(__file__).parent / 'data'
    N = np.load(data_dir / 'histprobs4.npy')   # shape (17, 23, 4)
    dx, dy = 1.5, 1.5
    xmin, xmax = -18.0, 18.0
    ymin, ymax = -12.0, 14.0
    x = np.arange(xmin, xmax, dx)[:-1] + dx / 2.0   # 23 points
    y = np.arange(ymin, ymax, dy)[:-1] + dy / 2.0   # 17 points
    _SPLINES = [RectBivariateSpline(y, x, N[:, :, n]) for n in range(4)]
    return _SPLINES


def getprobs(pc1, pc2):
    """Interpolate 4-state (E/H/T/N) probabilities from PC space.

    Parameters
    ----------
    pc1, pc2 : array_like, length M

    Returns
    -------
    list of M lists, each a 4-float (E, H, T, N) probability vector.
    """
    splines = _load_splines()
    pc1 = np.asarray(pc1, dtype=float)
    pc2 = np.asarray(pc2, dtype=float)
    probs = [[float(sp.ev(pc2[i], pc1[i])) for sp in splines]
             for i in range(len(pc1))]
    return probs


# ---------------------------------------------------------------------------
# Segment building helper for SS8 visualization
# ---------------------------------------------------------------------------

def _build_segments(s8_list, resi_list):
    """Build {ss_label: {start_resnum: width}} from aligned s8/resi lists."""
    segs: dict = {}
    if not s8_list:
        return segs
    prev = s8_list[0]
    start = resi_list[0]
    count = 1
    for i in range(1, len(s8_list)):
        if s8_list[i] == prev:
            count += 1
        else:
            segs.setdefault(prev, {})[start] = count
            prev = s8_list[i]
            start = resi_list[i]
            count = 1
    segs.setdefault(prev, {})[start] = count
    return segs


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_chezod_panel(ax, resi, zscores, colors, entry_id):
    """Colored Z-score bar chart."""
    for i, ri in enumerate(resi):
        zi = zscores[i]
        if np.isnan(zi):
            continue
        coli = tuple(float(c) for c in (colors[0][i], colors[1][i], colors[2][i]))
        ax.bar(ri - 0.5, zi, width=1.0, fc=coli, ec='none')
    ax.set_title(f'CheSPI color plot for: {entry_id}')
    ax.set_xlim(resi[0], resi[-1] + 1.0)
    ax.set_ylim(0, 16)
    ax.set_ylabel('CheZOD Z-score')


def _plot_populations_panel(ax, resi, probs, seq, entry_id):
    """Stacked bar chart of 4-state (EHTN) populations."""
    # reord=[1,2,3,0] converts EHTN → HTNE display order
    reord   = [1, 2, 3, 0]
    seccols = ('b', 'r', 'g', '0.5')   # EHTN colors
    for i, ri in enumerate(resi):
        probi = probs[i]
        ordered = [probi[n] for n in reord]   # HTNE
        acum = cumsum([0.0] + ordered)
        for n in range(4):
            ax.bar(ri - 0.5, ordered[n], bottom=acum[n], width=1.0,
                   fc=seccols[reord[n]], ec='none')
    ax.set_title(f'CheSPI populations for: {entry_id}')
    ax.set_xlim(resi[0], resi[-1] + 1.0)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability')


def _plot_ss8_panel(ax, s8_list, resi_list, probs8_list, entry_id):
    """SS8 segment diagram (top) + stacked probability bars (bottom half)."""
    import matplotlib.patches as mpatches

    SS8_COLS = {'H': 'r', 'G': 'm', 'I': 'w', 'T': 'g',
                'E': 'b', 'S': 'k', '-': '0.5', 'B': 'c'}

    segs = _build_segments(s8_list, resi_list)
    mini = resi_list[0]
    maxi = resi_list[-1]

    # Segment diagram (drawn at y ≈ 0.5)
    for ss in '-STGHIEB':
        if ss not in segs:
            continue
        col = SS8_COLS.get(ss, '0.5')
        for start, width in segs[ss].items():
            last = start + width
            if ss in 'EB':
                arrow = mpatches.Arrow(start, 0.5, width, 0.0,
                                       width=1.5, color='b')
                ax.add_patch(arrow)
            elif ss in 'GH':
                tn = 3.6 if ss == 'H' else 3.0
                n_turns = int(round((last - start) / tn * 2, 0)) / 2.0
                if n_turns < 0.5:
                    n_turns = 0.5
                x_val = linspace(0, n_turns * 2 * pi, 100)
                y_val = (-0.4 * np.sin(x_val) + 1) / 2
                x_val = x_val * (last - start) / (n_turns * 2 * pi) + start
                ax.plot(x_val, y_val, linewidth=3.0, c=col)
            elif ss == 'T':
                x_val = linspace(0, pi, 100)
                y_val = (0.3 * np.sin(x_val) + 1) / 2
                x_val = x_val / pi * width + start
                ax.plot(x_val, y_val, linewidth=2.0, c=col)
            elif ss in '-SI':
                lw = 2.0 if ss == 'S' else 1.5
                ax.plot([start, last], [0.5, 0.5], linewidth=lw, c=col)

    # Stacked probability bars (normalised to 0.0 – 0.5 range)
    allss8 = 'HGIE-TSB'
    cols8  = ['r', 'm', 'w', 'g', 'k', '0.5', 'c', 'b']
    reord8 = [0, 1, 2, 5, 6, 4, 7, 3]   # H,G,I,T,S,-,B,E
    SCALE  = 0.5   # squash probabilities into 0–0.5 y range

    for i, ri in enumerate(resi_list):
        pri_full = probs8_list[i]   # len-8 list in HGIE-TSB order
        fracs = [pri_full[reord8[k]] for k in range(8)]
        acum = cumsum([0.0] + fracs)
        for n in range(8):
            ax.bar(ri - 0.5, fracs[n] * SCALE, bottom=acum[n] * SCALE,
                   color=cols8[reord8[n]], width=1.0,
                   edgecolor=cols8[reord8[n]])

    ax.set_xlim(mini, maxi + 1.0)
    ax.set_ylim(0, 1)
    ax.set_title(f'CheSPI DSSP 8-class predictions for: {entry_id}')
    ax.set_ylabel('Probability')
    ax.set_xlabel('Residue number')


def _plot_2d_pcs_panel(ax, resi, pc1s, pc2s, colors, entry_id):
    """Scatter + line plot of PC1 vs PC2."""
    cols = array(getseccol(array(pc1s), array(pc2s))).T   # shape (N, 3)
    ax.scatter(pc1s, pc2s, c=cols, s=100)
    ax.plot(pc1s, pc2s, 'k')
    ax.axhline(0, color='k', linestyle='--')
    ax.axvline(0, color='k', linestyle='--')
    ax.set_xlim(-18, 18)
    ax.set_ylim(-16, 16)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(f'Principal components {entry_id}')


# ---------------------------------------------------------------------------
# Public plot function
# ---------------------------------------------------------------------------

def plot(outdir: Path, entry_id: str, result, ss_result=None,
         plot_2d: bool = False) -> None:
    """Render the CheSPI figure and save as PDF.

    Parameters
    ----------
    outdir     : output directory (Path)
    entry_id   : string identifier used in titles and filename
    result     : ChezodResult from chespi.chezod
    ss_result  : SSResult from chespi.prediction (None = skip panel 3)
    plot_2d    : if True, add a 2D PC1 vs PC2 scatter panel
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    resi   = result.residues
    zsco   = np.array(result.zscores, dtype=float)
    pc1s   = np.array(result.pc1, dtype=float)
    pc2s   = np.array(result.pc2, dtype=float)
    seq    = result.seq

    # Compute colors
    colors = getseccol(pc1s, pc2s)   # (r_arr, g_arr, b_arr)

    n_panels = 2 + (1 if ss_result is not None else 0) + (1 if plot_2d else 0)
    fig, axes = plt.subplots(n_panels, 1, figsize=(max(12, len(resi) * 0.12), 4 * n_panels))
    if n_panels == 1:
        axes = [axes]

    panel = 0

    # Panel 1: Z-score bars
    _plot_chezod_panel(axes[panel], resi, zsco, colors, entry_id)
    panel += 1

    # Panel 2: 4-state populations
    probs_4 = getprobs(pc1s, pc2s)
    _plot_populations_panel(axes[panel], resi, probs_4, seq, entry_id)
    panel += 1

    # Panel 3: SS8 (optional)
    if ss_result is not None:
        _plot_ss8_panel(axes[panel], ss_result.s8_list, ss_result.resi_list,
                        ss_result.probs8, entry_id)
        panel += 1

    # Panel 4: 2D PC scatter (optional)
    if plot_2d:
        _plot_2d_pcs_panel(axes[panel], resi, pc1s.tolist(), pc2s.tolist(),
                           colors, entry_id)
        panel += 1

    plt.tight_layout()
    out_path = Path(outdir) / f'cheSPIplot.pdf'
    plt.savefig(out_path)
    plt.close(fig)
    print(f'Plot saved → {out_path}')


# ---------------------------------------------------------------------------
# PyMOL structure coloring
# ---------------------------------------------------------------------------

def color_structure(pdb_path: str | Path, resi, colors,
                    outdir: Path, entry_id: str,
                    render: bool = False) -> None:
    """Write a PyMOL coloring script and optionally render a PNG.

    Parameters
    ----------
    pdb_path : path to PDB file
    resi     : sequence of residue numbers (1-indexed)
    colors   : sequence of (r, g, b) tuples in [0, 1]
    outdir   : output directory
    entry_id : string identifier
    render   : if True, launch headless PyMOL to ray-trace a PNG
               (requires pymol-open-source to be installed)
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pml_path = outdir / 'colCheSPI.pml'

    # Always write the PML script
    with open(pml_path, 'w') as f:
        for i, ri in enumerate(resi):
            r, g, b = (float(colors[0][i]), float(colors[1][i]), float(colors[2][i]))
            f.write(f'set_color coluser{ri}, [{r:.3f}, {g:.3f}, {b:.3f}]\n')
            f.write(f'color coluser{ri}, resi {ri}\n')

    print(f'PyMOL script written → {pml_path}')

    if render:
        png_path = outdir / f'structure_{entry_id}.png'
        try:
            import pymol
            pymol.finish_launching(['pymol', '-qc'])
            from pymol import cmd as pcmd
            pcmd.load(str(pdb_path))
            for i, ri in enumerate(resi):
                r, g, b = (float(colors[0][i]), float(colors[1][i]), float(colors[2][i]))
                pcmd.set_color(f'col{ri}', [r, g, b])
                pcmd.color(f'col{ri}', f'resi {ri}')
            pcmd.ray(1024, 768)
            pcmd.png(str(png_path))
            print(f'Structure PNG rendered → {png_path}')
        except ImportError:
            print('Warning: pymol-open-source not installed; '
                  'skipping render. Install with: pip install pymol-open-source')
        except Exception as exc:
            print(f'Warning: PyMOL rendering failed: {exc}')
