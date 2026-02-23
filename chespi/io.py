"""
io.py — NMR-STAR input via pynmrstar and all output file writing.

Output format
-------------
All write_* functions accept a ``fmt`` keyword (default ``'tsv'``):

    'tsv'   — tab-separated with a header row  (pandas default)
    'csv'   — comma-separated with a header row
    'space' — legacy fixed-width space-separated format (no header),
              identical to the CheSPI 1.x output for backward compatibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

try:
    import pynmrstar
except ImportError:
    raise ImportError("pynmrstar is required: pip install pynmrstar")

Fmt = Literal['tsv', 'csv', 'space']
_DEFAULT_FMT: Fmt = 'tsv'


# One-letter ↔ three-letter amino acid conversion
AA13 = {
    'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
    'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
    'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
    'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR',
}
AA31 = {v: k for k, v in AA13.items()}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_entry(input_str: str) -> pynmrstar.Entry:
    """Load from BMRB ID (int or numeric string) or local file path."""
    s = str(input_str).strip()
    if s.isdigit():
        return pynmrstar.Entry.from_database(int(s))
    return pynmrstar.Entry.from_file(s)


def get_entry_id(entry: pynmrstar.Entry) -> str:
    """Return a string identifier for the entry (BMRB ID or filename stem)."""
    try:
        return str(entry.entry_id)
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Sequence
# ---------------------------------------------------------------------------

def get_sequence(entry: pynmrstar.Entry) -> tuple[str, int]:
    """Return (one-letter sequence string, first residue number).

    Returns the sequence from the first entity saveframe that contains
    a polymer residue sequence.
    """
    for saveframe in entry.frame_list:
        # Look for entity saveframes with residue sequence
        try:
            loops = saveframe.loops
        except AttributeError:
            continue
        for loop in loops:
            tags = [t.lower() for t in loop.tags]
            if '_entity_comp_index.comp_id' in tags or 'comp_id' in tags:
                # Try pynmrstar style
                try:
                    id_col = None
                    comp_col = None
                    for t in loop.tags:
                        tl = t.lower()
                        if tl.endswith('.id') or tl == 'id':
                            id_col = t
                        if tl.endswith('.comp_id') or tl == 'comp_id':
                            comp_col = t
                    if comp_col is None:
                        continue
                    seq = ''
                    first = None
                    for row in loop:
                        comp = row[loop.tags.index(comp_col)].upper()
                        aa = AA31.get(comp)
                        if aa is None:
                            continue
                        seq += aa
                        if first is None and id_col is not None:
                            try:
                                first = int(row[loop.tags.index(id_col)])
                            except (ValueError, IndexError):
                                first = 1
                    if len(seq) >= 3:
                        return seq, (first if first is not None else 1)
                except Exception:
                    continue

    # Fallback: use pynmrstar's built-in entity sequence extraction
    for saveframe in entry.frame_list:
        try:
            cat = saveframe.category
            if 'entity' not in cat.lower():
                continue
            seq_tag = saveframe.get_tag('Mol_residue_sequence')
            if seq_tag:
                raw = seq_tag[0].replace('\n', '').replace(' ', '').upper()
                seq = ''
                for c in raw:
                    if c in AA13:
                        seq += c
                if len(seq) >= 3:
                    return seq, 1
        except Exception:
            continue

    raise ValueError("Could not extract sequence from NMR-STAR entry")


# ---------------------------------------------------------------------------
# Chemical shifts
# ---------------------------------------------------------------------------

def get_chemical_shifts(entry: pynmrstar.Entry) -> dict:
    """Return nested dict: {seq_id_str: {atom_id: [value, error, residue_label, ambc]}}.

    seq_id_str is the string representation of the sequence position (1-indexed).
    Shifts with error > 1.3 ppm are excluded (likely mis-assigned).
    """
    result: dict[str, dict[str, list]] = {}
    counts = {'H': 0, 'C': 0, 'N': 0, 'P': 0}

    for saveframe in entry.frame_list:
        try:
            cat = saveframe.category
            if 'assigned_chemical_shifts' not in cat.lower():
                continue
        except Exception:
            continue

        for loop in saveframe.loops:
            tags_lower = [t.lower() for t in loop.tags]

            # Detect column indices — pynmrstar may strip the category prefix
            # so 'Seq_ID' and '_atom_chem_shift.Seq_ID' both need to match.
            def find(suffix):
                bare = suffix.lstrip('.')
                for i, t in enumerate(tags_lower):
                    if t == bare or t.endswith(suffix):
                        return i
                return None

            seq_i   = find('.seq_id')
            comp_i  = find('.comp_id')
            atom_i  = find('.atom_id')
            type_i  = find('.atom_type')
            val_i   = find('.val')
            err_i   = find('.val_err')
            amb_i   = find('.ambiguity_code')

            if seq_i is None or val_i is None or atom_i is None:
                continue

            for row in loop:
                try:
                    seq_id = str(row[seq_i]).strip()
                    atom_id = str(row[atom_i]).strip()
                    val = float(row[val_i])
                    comp_id = str(row[comp_i]).upper() if comp_i is not None else 'UNK'
                    elem = str(row[type_i]).strip() if type_i is not None else atom_id[0]
                    ambc = str(row[amb_i]).strip() if amb_i is not None else '1'
                    err_raw = str(row[err_i]).strip() if err_i is not None else '.'
                    err = None if err_raw in ('.', '', '@') else float(err_raw)

                    # Exclude high-error shifts
                    if err is not None and err > 1.3:
                        continue

                    if seq_id not in result:
                        result[seq_id] = {}
                    result[seq_id][atom_id] = [val, err, comp_id, ambc]
                    if elem in counts:
                        counts[elem] += 1
                except (ValueError, IndexError, TypeError):
                    continue

    if not result:
        raise ValueError("No chemical shifts found in NMR-STAR entry")

    # Attach counts as attribute for downstream use
    result_obj = _ShiftStore(result)
    result_obj.counts = counts
    return result_obj


class _ShiftStore(dict):
    """dict subclass that also carries an atom-type count dict."""
    counts: dict


# ---------------------------------------------------------------------------
# Sample conditions
# ---------------------------------------------------------------------------

def get_sample_conditions(entry: pynmrstar.Entry) -> dict:
    """Return {'pH': float, 'temperature': float, 'ion': float}.

    Provides defaults (pH 7.0, 298 K, 0.1 M) when values are absent.
    """
    cond = {'pH': 7.0, 'temperature': 298.0, 'ion': 0.1}

    for saveframe in entry.frame_list:
        try:
            cat = saveframe.category
            if 'sample_conditions' not in cat.lower():
                continue
        except Exception:
            continue

        for loop in saveframe.loops:
            tags_lower = [t.lower() for t in loop.tags]

            type_i  = None
            val_i   = None
            unit_i  = None

            for i, t in enumerate(tags_lower):
                if t.endswith('.type') or t in ('variable_type', 'type'):
                    type_i = i
                if t.endswith('.val') or t in ('variable_value', 'val'):
                    # avoid matching val_err / val_units
                    if 'err' not in t and 'units' not in t:
                        val_i = i
                if t.endswith('.val_units') or t in ('variable_value_units', 'val_units'):
                    unit_i = i

            if type_i is None or val_i is None:
                continue

            for row in loop:
                try:
                    vtype = str(row[type_i]).strip().lower().strip("'")
                    vval  = str(row[val_i]).strip()
                    vunit = str(row[unit_i]).strip().lower() if unit_i is not None else ''
                    if vval in ('.', ''):
                        continue
                    fval = float(vval)

                    if 'ph' in vtype:
                        cond['pH'] = fval
                    elif 'temperature' in vtype:
                        cond['temperature'] = fval
                    elif 'ionic' in vtype or 'ion' in vtype:
                        if 'mm' in vunit:
                            fval /= 1000.0
                        if fval == 0:
                            fval = 0.001
                        cond['ion'] = fval
                except (ValueError, IndexError, TypeError):
                    continue

        # Also check scalar tags in the saveframe
        for tag_name in ('pH', 'Temp', 'Temperature', 'Ionic_strength'):
            try:
                vals = saveframe.get_tag(tag_name)
                if vals and vals[0] not in ('.', ''):
                    if 'ph' in tag_name.lower():
                        cond['pH'] = float(vals[0])
                    elif 'temp' in tag_name.lower():
                        cond['temperature'] = float(vals[0])
                    elif 'ion' in tag_name.lower():
                        cond['ion'] = float(vals[0])
            except Exception:
                continue

    return cond


def get_physical_state(entry: pynmrstar.Entry) -> str:
    """Return the physical state string (e.g. 'native', 'micelle', ...)."""
    for saveframe in entry.frame_list:
        try:
            vals = saveframe.get_tag('System_physical_state')
            if vals and vals[0] not in ('.', ''):
                return vals[0].strip("'").strip()
        except Exception:
            continue
    return 'native'


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _ensure(outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _save(df: pd.DataFrame, path: Path, fmt: Fmt) -> None:
    """Write *df* to *path* in the requested format."""
    if fmt == 'csv':
        df.to_csv(path, index=False)
    elif fmt == 'space':
        # Legacy fixed-width space-separated, no header
        with open(path, 'w') as f:
            for row in df.itertuples(index=False):
                f.write('  '.join(str(v) for v in row) + '\n')
    else:   # 'tsv' (default)
        df.to_csv(path, sep='\t', index=False)


def write_shifts(outdir: Path, entry_id: str, shift_records,
                 fmt: Fmt = _DEFAULT_FMT) -> None:
    """Write shifts: resnum atom observed predicted diff."""
    path = _ensure(outdir) / 'shifts.txt'
    df = pd.DataFrame(shift_records, columns=['resnum', 'atom', 'observed', 'predicted', 'diff'])
    if fmt == 'space':
        with open(path, 'w') as f:
            for rec in shift_records:
                f.write('%3d %2s %7.3f %5s %6.3f\n' % rec)
    else:
        _save(df, path, fmt)


def write_zscores(outdir: Path, entry_id: str,
                  seq: str, residues, zscores, pc1s, pc2s, pc3s, mini: int,
                  fmt: Fmt = _DEFAULT_FMT) -> None:
    """Write CheZOD/CheSPI components: aa resnum zscore pc1 pc2 pc3."""
    path = _ensure(outdir) / 'zscores.txt'
    rows = []
    for i, z in enumerate(zscores):
        if z < 99:
            I = i + mini
            aa = seq[I] if I < len(seq) else 'X'
            rows.append((aa, I + 1, round(float(z), 3),
                         round(float(pc1s[i]), 3),
                         round(float(pc2s[i]), 3),
                         round(float(pc3s[i]), 3)))
    df = pd.DataFrame(rows, columns=['aa', 'resnum', 'zscore', 'pc1', 'pc2', 'pc3'])
    if fmt == 'space':
        with open(path, 'w') as f:
            for row in rows:
                f.write('%s %3d %6.3f %6.3f %6.3f %6.3f\n' % row)
    else:
        _save(df, path, fmt)


def write_colors(outdir: Path, entry_id: str, resi, colors,
                 fmt: Fmt = _DEFAULT_FMT) -> None:
    """Write residue colors: resnum hex r g b."""
    path = _ensure(outdir) / 'colors.txt'
    rows = []
    for i, ri in enumerate(resi):
        rgb = tuple(int(c * 255) for c in colors[i])
        hexstr = '#%02x%02x%02x' % rgb
        rows.append((ri, hexstr, rgb[0], rgb[1], rgb[2]))
    df = pd.DataFrame(rows, columns=['resnum', 'hex', 'r', 'g', 'b'])
    if fmt == 'space':
        with open(path, 'w') as f:
            for row in rows:
                f.write('%3d %s %3d %3d %3d\n' % row)
    else:
        _save(df, path, fmt)


def write_populations(outdir: Path, entry_id: str, seq: str, resi, probs,
                      fmt: Fmt = _DEFAULT_FMT) -> None:
    """Write 3-state (HTNE) populations: aa resnum H T N E."""
    path = _ensure(outdir) / 'populations.txt'
    reord = [1, 2, 3, 0]   # HTNE order
    rows = []
    for i, ri in enumerate(resi):
        probi = probs[i]
        aa = seq[ri - 1] if ri - 1 < len(seq) else 'X'
        h, t, n, e = [round(float(probi[n]), 4) for n in reord]
        rows.append((aa, ri, h, t, n, e))
    df = pd.DataFrame(rows, columns=['aa', 'resnum', 'H', 'T', 'N', 'E'])
    if fmt == 'space':
        with open(path, 'w') as f:
            for row in rows:
                f.write('%s %3d %5.3f %5.3f %5.3f %5.3f\n' % row)
    else:
        _save(df, path, fmt)


def write_probs8(outdir: Path, entry_id: str, seq, resi_list, probs_list,
                 fmt: Fmt = _DEFAULT_FMT) -> None:
    """Write 8-state probabilities: aa resnum H G I E - T S B."""
    path = _ensure(outdir) / 'probs8.txt'
    cols = ['aa', 'resnum', 'p_H', 'p_G', 'p_I', 'p_E', 'p_-', 'p_T', 'p_S', 'p_B']
    rows = []
    for seq_i, ri, p in zip(seq, resi_list, probs_list):
        rows.append((seq_i, ri) + tuple(round(float(x), 4) for x in p))
    df = pd.DataFrame(rows, columns=cols)
    if fmt == 'space':
        with open(path, 'w') as f:
            for row in rows:
                f.write(' %3s %3d' % row[:2] +
                        (' %6.4f' * 8) % row[2:] + '\n')
    else:
        _save(df, path, fmt)


def write_probs3(outdir: Path, entry_id: str, seq, resi_list, probs3_list,
                 fmt: Fmt = _DEFAULT_FMT) -> None:
    """Write 3-state probabilities: aa resnum p_H p_S p_C."""
    path = _ensure(outdir) / 'probs3.txt'
    rows = []
    for seq_i, ri, p in zip(seq, resi_list, probs3_list):
        rows.append((seq_i, ri, round(float(p[0]), 4),
                     round(float(p[1]), 4), round(float(p[2]), 4)))
    df = pd.DataFrame(rows, columns=['aa', 'resnum', 'p_H', 'p_S', 'p_C'])
    if fmt == 'space':
        with open(path, 'w') as f:
            for row in rows:
                f.write(' %3s %3d %6.4f %6.4f %6.4f\n' % row)
    else:
        _save(df, path, fmt)


def write_max8(outdir: Path, entry_id: str, seq, resi_list,
               s8_list, prob_list, post_list,
               fmt: Fmt = _DEFAULT_FMT) -> None:
    """Write best 8-state prediction: aa resnum ss8 prob post."""
    path = _ensure(outdir) / 'max8.txt'
    rows = [(seq_i, ri, s8, round(float(pr), 4), round(float(po), 4))
            for seq_i, ri, s8, pr, po in zip(seq, resi_list, s8_list, prob_list, post_list)]
    df = pd.DataFrame(rows, columns=['aa', 'resnum', 'ss8', 'prob', 'post'])
    if fmt == 'space':
        with open(path, 'w') as f:
            for row in rows:
                f.write(' %3s %3d %s %6.4f %6.4f\n' % row)
    else:
        _save(df, path, fmt)


def write_max3(outdir: Path, entry_id: str, seq, resi_list,
               s3_list, prob_list, post_list,
               fmt: Fmt = _DEFAULT_FMT) -> None:
    """Write best 3-state prediction: aa resnum ss3 prob post."""
    path = _ensure(outdir) / 'max3.txt'
    rows = [(seq_i, ri, s3, round(float(pr), 4), round(float(po), 4))
            for seq_i, ri, s3, pr, po in zip(seq, resi_list, s3_list, prob_list, post_list)]
    df = pd.DataFrame(rows, columns=['aa', 'resnum', 'ss3', 'prob', 'post'])
    if fmt == 'space':
        with open(path, 'w') as f:
            for row in rows:
                f.write(' %3s %3d %s %6.4f %6.4f\n' % row)
    else:
        _save(df, path, fmt)


def write_summary8(outdir: Path, entry_id: str,
                   seq_str: str, conf_str: str, s8_str: str) -> None:
    """Write human-readable 8-state one-liner summary (always fixed format)."""
    N = len(s8_str)
    dnums = '1234567890'
    dtens, dones = divmod(N, 10)
    dstr = dtens * dnums + dnums[:dones]
    with open(_ensure(outdir) / 'summary8.txt', 'w') as f:
        f.write(dstr + '\n' + seq_str + '\n' + conf_str + '\n' + s8_str + '\n')


def write_summary3(outdir: Path, entry_id: str,
                   seq_str: str, conf_str: str, s3_str: str) -> None:
    """Write human-readable 3-state one-liner summary (always fixed format)."""
    N = len(s3_str)
    dnums = '1234567890'
    dtens, dones = divmod(N, 10)
    dstr = dtens * dnums + dnums[:dones]
    with open(_ensure(outdir) / 'summary3.txt', 'w') as f:
        f.write(dstr + '\n' + seq_str + '\n' + conf_str + '\n' + s3_str + '\n')


def write_pml_script(outdir: Path, entry_id: str, resi, colors) -> None:
    """Write PyMOL coloring script (always fixed format)."""
    with open(_ensure(outdir) / 'colCheSPI.pml', 'w') as f:
        for i, ri in enumerate(resi):
            rgb = colors[i]
            f.write('set_color coluser%d, [%5.3f, %5.3f, %5.3f]\n' % (ri, *rgb))
            f.write('color coluser%d, resi %s\n' % (ri, ri))
