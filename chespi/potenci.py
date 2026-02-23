"""
potenci.py — POTENCI random-coil chemical shift prediction.

Ported from chespi_for_nmrstar31.py (original by Jakob Toudahl Nielsen
and Frans A A Mulder, Mulder Laboratory).

Public interface:
    predict(seq, pH, temperature, ion) -> dict
        Returns {(resnum, aa): {atom: shift_value, ...}, ...}
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.special import erfc
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).parent / "data"

# ---------------------------------------------------------------------------
# Verbose flag (set True for debugging)
# ---------------------------------------------------------------------------
VERB = False

# Standard amino acids supported
AAstandard = 'ACDEFGHIKLMNPQRSTVWY'

# ---------------------------------------------------------------------------
# Table loading helpers
# ---------------------------------------------------------------------------

def _read(fname: str) -> list[str]:
    return (Path(_DATA_DIR) / fname).read_text().splitlines()


def _initcorcents() -> dict:
    lines = [l for l in _read("tablecent.csv") if l.strip()]
    headers = lines[0].split()[1:]   # atom names (C CA CB N H HA HB)
    dct: dict = {}
    for line in lines[1:]:
        vals = line.split()
        aa = vals[0]
        dct[aa] = {}
        for j, atn in enumerate(headers):
            raw = vals[1 + j]
            dct[aa][atn] = None if raw == 'None' else float(raw)
    return dct


def _initcorneis() -> dict:
    lines = [l for l in _read("tablenei.csv") if l.strip()]
    dct: dict = {}
    for line in lines:
        vals = line.split()
        atn, aa = vals[0], vals[1]
        if aa not in dct:
            dct[aa] = {}
        dct[aa][atn] = [float(vals[2 + j]) for j in range(4)]
    # terminal corrections
    for line in _read("tabletermcorrs.csv"):
        vals = line.split()
        if len(vals) < 3:
            continue
        atn, term = vals[0], vals[1]
        if term not in dct:
            dct[term] = {}
        if term == 'n':
            dct['n'][atn] = [None, None, None, float(vals[-1])]
        elif term == 'c':
            dct['c'][atn] = [float(vals[-1]), None, None, None]
    return dct


def _gettempkoeff() -> dict:
    lines = [l for l in _read("tabletempk.csv") if l.strip()]
    headers = lines[0].split()[1:]
    dct: dict = {atn: {} for atn in headers}
    for line in lines[1:]:
        vals = line.split()
        aa = vals[0]
        for j, atn in enumerate(headers):
            dct[atn][aa] = float(vals[1 + j])
    return dct


def _initcorrcomb() -> dict:
    dct: dict = {}
    for line in _read("tablecombdevs.csv"):
        vals = line.split()
        if len(vals) < 7:
            continue
        atn = vals[0]
        if atn not in dct:
            dct[atn] = {}
        neipos    = int(vals[1])
        centgroup = vals[2]
        neigroup  = vals[3]
        segment   = vals[4]
        dct[atn][segment] = ((neipos, centgroup, neigroup), float(vals[-2]))
    return dct


# Module-level precomputed tables
CENTSHIFTS = _initcorcents()
NEICORRS   = _initcorneis()
COMBCORRS  = _initcorrcomb()
TEMPCORRS  = _gettempkoeff()


# ---------------------------------------------------------------------------
# Shift prediction functions
# ---------------------------------------------------------------------------

def predPentShift(pent: str, atn: str):
    """Predict chemical shift from pentamer sequence context."""
    aac = pent[2]
    sh = CENTSHIFTS[aac][atn]
    if sh is None:
        return None
    allneipos = [2, 1, -1, -2]
    for i in range(4):
        aai = pent[2 + allneipos[i]]
        if aai in NEICORRS:
            corr = NEICORRS[aai][atn][i]
            if corr is not None:
                sh += corr
    groups = ['G', 'P', 'FYW', 'LIVMCA', 'KR', 'DE']
    labels = 'GPra+-p'
    grstr = ''
    for i in range(5):
        aai = pent[i]
        found = False
        for j, gr in enumerate(groups):
            if aai in gr:
                grstr += labels[j]
                found = True
                break
        if not found:
            grstr += 'p'
    centgr = grstr[2]
    if atn in COMBCORRS:
        for segm in COMBCORRS[atn]:
            key, combval = COMBCORRS[atn][segm]
            neipos, centgroup, neigroup = key
            if centgroup == centgr and grstr[2 + neipos] == neigroup:
                if (centgr, neigroup) != ('p', 'p') or pent[2] in 'ST':
                    sh += combval
    return sh


def gettempcorr(aai: str, atn: str, temp: float) -> float:
    return TEMPCORRS[atn][aai] / 1000.0 * (temp - 298)


# ---------------------------------------------------------------------------
# pKa / pH correction (PePKalc)
# ---------------------------------------------------------------------------

_R = 8.314472
_e = 79.0
_a = 5.0
_b = 7.5
_cutoff = 2
_ncycles = 5

pK0 = {
    "n": 8.23, "D": 3.86, "E": 4.34, "H": 6.45,
    "C": 8.49, "K": 10.34, "R": 13.9, "Y": 9.76, "c": 3.55,
}


def fun(pH: float, pK: float, nH: float) -> float:
    return (10 ** (nH * (pK - pH))) / (1.0 + (10 ** (nH * (pK - pH))))


def _W(r, Ion=0.1):
    k = np.sqrt(Ion) / 3.08
    x = k * r / np.sqrt(6)
    return 332.286 * np.sqrt(6 / np.pi) * (1 - np.sqrt(np.pi) * x * np.exp(x ** 2) * erfc(x)) / (_e * r)


def _w2logp(x, T=293.15):
    return x * 4181.2 / (_R * T * np.log(10))


def _smallmatrixlimits(ires, cutoff, length):
    ileft = max(1, ires - cutoff)
    iright = min(ileft + 2 * cutoff, length)
    if iright == length:
        ileft = max(1, iright - 2 * cutoff)
    return ileft, iright


def _smallmatrixpos(ires, cutoff, length):
    resi = cutoff + 1
    if ires < cutoff + 1:
        resi = ires
    if ires > length - cutoff:
        resi = min(length, 2 * cutoff + 1) - (length - ires)
    return resi


def calc_pkas_from_seq(seq: str, T: float = 293.15, Ion: float = 0.1) -> dict:
    """Calculate pKa values for titratable groups in sequence using PePKalc."""
    pHs = np.arange(1.99, 10.01, 0.15)
    pos = np.array([i for i in range(len(seq)) if seq[i] in pK0])
    if len(pos) == 0:
        return {}
    N = pos.shape[0]
    I = np.diag(np.ones(N))
    sites = ''.join([seq[i] for i in pos])
    neg = np.array([i for i in range(len(sites)) if sites[i] in 'DEYc'])
    l = np.array([abs(pos - pos[i]) for i in range(N)])
    d = _a + np.sqrt(l) * _b
    tmp = _W(d, Ion)
    tmp[I == 1] = 0
    ww = _w2logp(tmp, T) / 2
    chargesempty = np.zeros(pos.shape[0])
    if len(neg):
        chargesempty[neg] = -1
    pK0s = [pK0[c] for c in sites]
    nH0s = [0.9 for c in sites]
    titration = np.zeros((N, len(pHs)))
    smallN = min(2 * _cutoff + 1, len(pos))
    smallI = np.diag(np.ones(smallN))
    alltuples = [[int(c) for c in np.binary_repr(i, smallN)] for i in range(2 ** smallN)]
    gmatrix = [np.zeros((smallN, smallN)) for _ in range(len(pHs))]
    for icycle in range(_ncycles):
        if icycle == 0:
            fractionhold = np.array([[fun(pHs[p], pK0s[i], nH0s[i])
                                      for i in range(N)] for p in range(len(pHs))])
        else:
            fractionhold = titration.transpose()
        for ires in range(1, N + 1):
            ileft, iright = _smallmatrixlimits(ires, _cutoff, N)
            resi = _smallmatrixpos(ires, _cutoff, N)
            for p in range(len(pHs)):
                fraction = fractionhold[p].copy()
                fraction[ileft - 1:iright] = 0
                charges = chargesempty + fraction
                ww0 = np.diag(np.dot(ww, charges) * 2)
                gmatrixfull = ww + ww0 + pHs[p] * I - np.diag(pK0s)
                gmatrix[p] = gmatrixfull[ileft - 1:iright, ileft - 1:iright]
            E_all = np.array([sum([10 ** -(gmatrix[p] * np.outer(c, c)).sum()
                                   for c in alltuples]) for p in range(len(pHs))])
            E_sel = np.array([sum([10 ** -(gmatrix[p] * np.outer(c, c)).sum()
                                   for c in alltuples if c[resi - 1] == 1])
                              for p in range(len(pHs))])
            titration[ires - 1] = E_sel / E_all
        sol = np.array([curve_fit(fun, pHs, titration[p], [pK0s[p], nH0s[p]])[0]
                        for p in range(len(pK0s))])
        pKs, nHs = sol.transpose()
    dct = {}
    for p, i in enumerate(pos):
        dct[i - 1] = (pKs[p], nHs[p], seq[i])
    return dct


def _get_phshifts() -> dict:
    """Parse tablephshifts into a nested dict {residue: {atom: delta_shift}}."""
    dct: dict = {}
    for line in _read("tablephshifts.csv"):
        vals = line.split()
        if len(vals) > 3:
            resn, atn = vals[0], vals[1]
            try:
                shd = float(vals[4])
            except (IndexError, ValueError):
                continue
            if resn not in dct:
                dct[resn] = {}
            dct[resn][atn] = shd
            if len(vals) > 6:
                for n in range(2):
                    try:
                        shdn = float(vals[5 + n])
                    except (IndexError, ValueError):
                        continue
                    nresn = resn + 'ps'[n]
                    if nresn not in dct:
                        dct[nresn] = {}
                    dct[nresn][atn] = shdn
    return dct


# In-memory pKa cache (keyed by (seq, temperature, ion))
_pka_cache: dict = {}
_phshifts_cache: dict | None = None


def getphcorrs(seq: str, temperature: float, pH: float, ion: float) -> dict:
    """Compute per-residue pH correction offsets for each backbone atom."""
    global _phshifts_cache
    if _phshifts_cache is None:
        _phshifts_cache = _get_phshifts()
    bbatns = ['C', 'CA', 'CB', 'HA', 'H', 'N', 'HB']
    phshifts = _phshifts_cache
    Ion = max(0.0001, ion)
    cache_key = (seq[:150], temperature, Ion)
    if cache_key in _pka_cache:
        pkadct = _pka_cache[cache_key]
    else:
        pkadct = calc_pkas_from_seq('n' + seq + 'c', temperature, Ion)
        _pka_cache[cache_key] = pkadct
    outdct: dict = {}
    for i in pkadct:
        pKa, nH, resi = pkadct[i]
        frac  = fun(pH,  pKa,       nH)
        frac7 = fun(7.0, pK0[resi], nH)
        if resi in 'nc':
            continue
        for atn in bbatns:
            if atn not in outdct:
                outdct[atn] = {}
            dctresi = phshifts.get(resi, {})
            try:
                delta = dctresi[atn]
                jump  = frac  * delta
                jump7 = frac7 * delta
            except KeyError:
                continue
            if abs(delta) < 99:
                jumpdelta = jump - jump7
                if i not in outdct[atn]:
                    outdct[atn][i] = [resi, jumpdelta]
                else:
                    outdct[atn][i][0] = resi
                    outdct[atn][i][1] += jumpdelta
                nresn_p = resi + 'p'
                if nresn_p in phshifts and atn in phshifts[nresn_p]:
                    for n in range(2):
                        ni = i + 2 * n - 1
                        nresn = resi + 'ps'[n]
                        ndelta = phshifts[nresn][atn]
                        njump  = frac  * ndelta
                        njump7 = frac7 * ndelta
                        njumpdelta = njump - njump7
                        if ni not in outdct[atn]:
                            outdct[atn][ni] = [None, njumpdelta]
                        else:
                            outdct[atn][ni][1] += njumpdelta
    return outdct


def getpredshifts(seq: str, temperature: float, pH: float, ion: float,
                  usephcor: bool = True) -> dict:
    """Generate POTENCI random-coil shift predictions for a sequence.

    Returns {(resnum, aa): {atom: shift_value, ...}}
    """
    bbatns = ['C', 'CA', 'CB', 'HA', 'H', 'N', 'HB']
    phcorrs = getphcorrs(seq, temperature, pH, ion) if usephcor else {}
    shiftdct: dict = {}
    for i in range(1, len(seq) - 1):
        if seq[i] in AAstandard:
            trip = seq[i - 1] + seq[i] + seq[i + 1]
            shiftdct[(i + 1, seq[i])] = {}
            for at in bbatns:
                if (trip[1], at) in [('G', 'CB'), ('G', 'HB'), ('P', 'H')]:
                    continue
                if i == 1:
                    pent = 'n' + trip + seq[i + 2]
                elif i == len(seq) - 2:
                    pent = seq[i - 2] + trip + 'c'
                else:
                    pent = seq[i - 2] + trip + seq[i + 2]
                shp = predPentShift(pent, at)
                if shp is not None:
                    if not (at in ('CA', 'CB') and seq[i] == 'C'):
                        if at != 'HB':
                            shp += gettempcorr(trip[1], at, temperature)
                        if at in phcorrs and i in phcorrs[at]:
                            phdata = phcorrs[at][i]
                            if abs(phdata[1]) < 9.9:
                                shp -= phdata[1]
                        shiftdct[(i + 1, seq[i])][at] = shp
    return shiftdct


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict(seq: str, pH: float = 7.0, temperature: float = 298.0,
            ion: float = 0.1) -> dict:
    """Predict POTENCI random-coil chemical shifts for the given sequence.

    Parameters
    ----------
    seq         : one-letter amino acid sequence string
    pH          : pH value (default 7.0)
    temperature : temperature in Kelvin (default 298 K)
    ion         : ionic strength in M (default 0.1 M)

    Returns
    -------
    dict: {(resnum, aa_letter): {atom: shift_ppm, ...}}
        resnum is 1-indexed, matching the sequence position.
    """
    if len(seq) < 5:
        raise ValueError("Sequence must have at least 5 residues for POTENCI")
    usephcor = pH < 6.99 or pH > 7.01
    return getpredshifts(seq, temperature, pH, ion, usephcor)
