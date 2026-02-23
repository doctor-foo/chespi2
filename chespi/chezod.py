"""
chezod.py — CheZOD Z-scores and CheSPI principal components computation.

Ported from chespi_for_nmrstar31.py (ShiftGetter.cmp2pred1,
ShiftGetter.visresults, getCheZODandPCs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy import sqrt, array, average, zeros, std, log, exp, clip, cumsum


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ChezodResult:
    """Container for CheZOD Z-scores and CheSPI principal components."""
    seq: str                    # full sequence
    mini: int                   # 0-indexed minimum position (= first residue index)
    residues: list              # 1-indexed residue numbers with data
    zscores: np.ndarray         # CheZOD Z-scores (NaN where no data)
    pc1: np.ndarray             # CheSPI PC1
    pc2: np.ndarray             # CheSPI PC2
    pc3: np.ndarray             # CheSPI PC3
    shift_records: list = field(default_factory=list)  # (resnum, atom, observed, pent, diff)


# Standard amino acids
_AA1S = sorted('ACDEFGHIKLMNPQRSTVWY')

# Backbone atoms for CheZOD computation
_BBATNS = ['C', 'CA', 'CB', 'HA', 'H', 'N', 'HB']

# Refined weights (from PCA calibration)
_REFINED_WEIGHTS = {
    'C': 0.1846, 'CA': 0.1982, 'CB': 0.1544,
    'HA': 0.02631, 'H': 0.06708, 'N': 0.4722, 'HB': 0.02154,
}

# Outlier thresholds (normalized units)
_OUTLI_VALS = {
    'C': 5.0, 'CA': 7.0, 'CB': 7.0,
    'HA': 1.80, 'H': 2.30, 'N': 12.00, 'HB': 1.80,
}

# PCA loadings (OPLS-DA weights for PC1, PC2, PC3)
_WBUF = [
    ['weights:', 'N',  '-0.0626', '0.0617',  '0.2635'],
    ['weights:', 'C',  '0.2717',  '0.2466',  '0.0306'],
    ['weights:', 'CA', '0.2586',  '0.2198',  '0.0394'],
    ['weights:', 'CB', '-0.2635', '0.1830',  '-0.1877'],
    ['weights:', 'H',  '-0.3620', '1.3088',  '0.3962'],
    ['weights:', 'HA', '-1.0732', '0.4440',  '-0.4673'],
    ['weights:', 'HB', '0.5743',  '0.2262',  '-0.3388'],
]
_WDCT = {lin[1]: [float(lin[n]) for n in (2, 3, 4)] for lin in _WBUF}


# ---------------------------------------------------------------------------
# Chi-squared → Z-score conversion
# ---------------------------------------------------------------------------

def convChi2CDF(rss: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Convert chi-squared statistics to approximate Z-scores."""
    r = rss / k
    num = ((r ** (1.0/6)) - 0.50 * (r ** (1.0/3)) + (1.0/3) * (r ** 0.5)) \
          - (5.0/6 - 1.0/9/k - 7.0/648/(k**2) + 25.0/2187/(k**3))
    den = sqrt(1.0/18/k + 1.0/162/(k**2) - 37.0/11664/(k**3))
    return num / den


# ---------------------------------------------------------------------------
# Compare observed vs predicted shifts
# ---------------------------------------------------------------------------

def _cmp2pred(obs_shifts: dict, pred_shifts: dict, seq: str) -> dict:
    """Compute per-residue, per-atom secondary shifts (obs - pred).

    Parameters
    ----------
    obs_shifts  : {seq_id_str: {atom: [val, err, comp, ambc]}}
    pred_shifts : {(resnum, aa): {atom: shift}}
    seq         : one-letter sequence

    Returns
    -------
    cmpdct : {atom: {seq_index_0based: delta_shift}}
    shiftdct : {(seq_index_0based, atom): [observed, pentamer_context]}
    """
    cmpdct: dict = {}
    shiftdct: dict = {}
    aa1s = _AA1S

    for i in range(1, len(seq) - 1):
        res = str(i + 1)
        if res not in obs_shifts:
            continue
        if seq[i] not in aa1s:
            continue
        shdct = obs_shifts[res]
        for at in _BBATNS:
            sho = None
            if at in shdct:
                sho = shdct[at][0]
            elif seq[i] == 'G' and at in ('HA', 'HB'):
                shs = []
                for pref in '23':
                    atp = at + pref
                    if atp in shdct:
                        shs.append(shdct[atp][0])
                if shs:
                    sho = float(np.mean(shs))
            if sho is None:
                continue
            # Look up predicted shift
            key = (i + 1, seq[i])
            if key not in pred_shifts:
                continue
            pred_dct = pred_shifts[key]
            if at not in pred_dct:
                continue
            if seq[i] == 'C' and at in ('CA', 'CB'):
                continue
            shp = pred_dct[at]
            shiftdct[(i, at)] = [sho, seq[i - 1:i + 2]]
            diff = sho - shp
            if at not in cmpdct:
                cmpdct[at] = {}
            cmpdct[at][i] = diff

    return cmpdct, shiftdct


# ---------------------------------------------------------------------------
# visresults equivalent — compute Z-scores and PCs (no plotting)
# ---------------------------------------------------------------------------

def _compute_zscores_and_pcs(
    cmpdct: dict,
    shiftdct: dict,
    seq: str,
    offdct: Optional[dict] = None,
    min_aic: float = 5.0,
    cdfthr: float = 6.0,
    return_offsets: bool = False,
    return_components: bool = False,
):
    """Core computation: chi-squared Z-scores + PC projections.

    When offdct is None  → estimate running offset and return it.
    When offdct is given → compute Z-scores/PCs using those offsets.
    """
    bbatns = _BBATNS
    wdct = _WDCT
    refined_weights = _REFINED_WEIGHTS
    outlivals = _OUTLI_VALS

    maxi = max(max(cmpdct[at].keys()) for at in cmpdct)
    mini = min(min(cmpdct[at].keys()) for at in cmpdct)
    nres = maxi - mini + 1
    resids = list(range(mini + 1, maxi + 2))

    tot        = zeros(nres)
    newtot     = zeros(nres)
    newtotsgn  = zeros(nres)
    newtotsgn1 = zeros(nres)
    newtotsgn2 = zeros(nres)
    totnum     = zeros(nres)
    allrmsd    = []
    allruns    = zeros(nres)
    rdct: dict = {}
    oldct: dict = {}
    dats: dict = {}

    for at in cmpdct:
        vol = outlivals[at]
        subtot  = zeros(nres)
        subtot1 = zeros(nres)
        subtot2 = zeros(nres)

        A = array(list(cmpdct[at].items()))  # shape (N, 2): [[index, diff], ...]
        w = refined_weights[at]
        shw = A[:, 1] / w
        off = average(shw)
        rms0 = sqrt(average(shw ** 2))

        if offdct is not None:
            shw -= offdct.get(at, 0.0)

        for i in range(len(A)):
            resi = int(A[i][0]) - mini
            ashwi = abs(shw[i])
            if ashwi > cdfthr:
                oldct[(at, resi)] = ashwi
            tot[resi] += (min(4.0, ashwi) ** 2)
            for k in [-1, 0, 1]:
                if 0 <= resi + k < len(subtot):
                    subtot [resi + k] += clip(shw[i] * w, -vol, vol) * wdct[at][0]
                    subtot1[resi + k] += clip(shw[i] * w, -vol, vol) * wdct[at][1]
                    subtot2[resi + k] += clip(shw[i] * w, -vol, vol) * wdct[at][2]
            totnum[resi] += 1
            # Running offset estimation (only when offdct is None)
            if offdct is None:
                if 3 < i < len(A) - 4:
                    vals = shw[i - 4:i + 5]
                    runstd = std(vals)
                    allruns[resi] += runstd
                    if resi not in rdct:
                        rdct[resi] = {}
                    rdct[resi][at] = average(vals), sqrt(average(vals ** 2)), runstd

        dats[at] = shw
        stdw = std(shw)
        dAIC = log(rms0 / stdw) * len(A) - 1 if stdw > 0 else 0
        allrmsd.append(stdw)

        newtot     += ((subtot  / 3.0) ** 2)
        newtotsgn  += subtot
        newtotsgn1 += subtot1
        newtotsgn2 += subtot2

    # Tri-residue smoothing for Z-scores
    T0  = list(tot / np.where(totnum > 0, totnum, 1))
    Th  = list(tot / np.where(totnum > 0, totnum, 1) * 0.5)
    Ts  = list(tot)
    Tn  = list(totnum)
    tot3f  = array([0, 0] + Ts) + array([0] + Ts + [0]) + array(Ts + [0, 0])
    totn3f = array([0, 0] + Tn) + array([0] + Tn + [0]) + array(Tn + [0, 0])
    cdfs3  = convChi2CDF(tot3f[1:-1], np.where(totn3f[1:-1] > 0, totn3f[1:-1], 1))

    # Replace invalid (totnum==0) entries with NaN
    cdfs3 = np.where(totn3f[1:-1] > 0, cdfs3, float('nan'))

    # Running offset estimation
    if offdct is None:
        tr = (allruns / np.where(totnum > 0, totnum, 1))[4:-4]
        offdct_out = {}
        mintr = None
        minval = 999
        for j in range(len(tr)):
            if j + 4 in rdct and len(rdct[j + 4]) == len(cmpdct):
                if tr[j] < minval:
                    minval = tr[j]
                    mintr = j
        if mintr is None:
            return None  # cannot estimate offset
        for at in rdct[mintr + 4]:
            roff, std0, stdc = rdct[mintr + 4][at]
            dAIC = log(std0 / stdc) * 9 - 1 if stdc > 0 else 0
            if dAIC > min_aic:
                offdct_out[at] = roff
                print(f'  offset correction accepted: {at} {roff:.4f} dAIC={dAIC:.2f}')
            else:
                print(f'  offset correction rejected (low dAIC): {at} {roff:.4f} dAIC={dAIC:.2f}')
                offdct_out[at] = 0.0
        return offdct_out

    if return_offsets:
        # Second pass: compute accepted offsets
        atns = list(cmpdct.keys())
        accdct = {at: [] for at in atns}
        numol = 0
        aresids = array(resids)

        # Identify outliers
        cdfs = convChi2CDF(tot / np.where(totnum > 0, totnum, 1),
                           np.where(totnum > 0, totnum, 1))
        numzslt3 = int(np.sum(cdfs3[~np.isnan(cdfs3)] < cdfthr))
        finaloutli = [i + mini + 1 for i in range(nres)
                      if (not np.isnan(cdfs[i]) and cdfs[i] > cdfthr) or
                         (not np.isnan(cdfs3[i]) and cdfs3[i] > cdfthr and
                          totnum[i] > 0)]

        iatns = sorted(shiftdct.keys())
        for i_at in iatns:
            i, at = i_at
            w = refined_weights[at]
            ol = False
            if i + 1 in finaloutli:
                ol = True
            elif (at, i - mini) in oldct:
                ol = True
            if not ol:
                accdct[at].append(cmpdct[at][i])
            else:
                numol += 1

        newoffdct = {}
        sumrmsd = 0.0
        totsh = 0
        for at in accdct:
            w = refined_weights[at]
            vals = array(accdct[at]) / w
            anum = len(vals)
            if anum == 0:
                newoffdct[at] = 0.0
            else:
                aoff  = average(vals)
                astd0 = sqrt(average(vals ** 2))
                astdc = std(vals) if len(vals) > 1 else astd0
                adAIC = log(astd0 / astdc) * anum - 1 if astdc > 0 and astd0 > 0 else 0
                if adAIC < min_aic or anum < 4:
                    print(f'  final offset rejected: {at} {aoff:.4f} dAIC={adAIC:.2f}')
                    astdc = astd0
                    aoff = 0.0
                else:
                    print(f'  final offset accepted: {at} {aoff:.4f} dAIC={adAIC:.2f}')
                sumrmsd += astdc * anum
                totsh += anum
                newoffdct[at] = aoff

        avewrmsd = sumrmsd / totsh if totsh > 0 else 9.99
        fracacc  = totsh / (totsh + numol) if (totsh + numol) > 0 else 0.0
        return avewrmsd, fracacc, newoffdct, cdfs3

    # return_components=True: return per-residue Z-scores and PCs
    if return_components:
        pc1ws = newtotsgn  / np.where(totn3f[1:-1] > 0, sqrt(totn3f[1:-1]), 1) * 8.0
        pc2ws = newtotsgn1 / np.where(totn3f[1:-1] > 0, sqrt(totn3f[1:-1]), 1) * 8.0
        pc3ws = newtotsgn2 / np.where(totn3f[1:-1] > 0, sqrt(totn3f[1:-1]), 1) * 8.0
        return resids, cdfs3, pc1ws, pc2ws, pc3ws

    # Default: return cdfs3, allrmsd
    avc = average(cdfs3[~np.isnan(cdfs3) & (cdfs3 < 20.0)]) if np.any(~np.isnan(cdfs3)) else 0
    return average(allrmsd), avc, cdfs3


# ---------------------------------------------------------------------------
# Shift record extraction (for output)
# ---------------------------------------------------------------------------

def _extract_shift_records(cmpdct, shiftdct):
    """Return list of (resnum, atom, observed, pent, diff) tuples for output."""
    records = []
    iatns = sorted(shiftdct.keys())
    for i, at in iatns:
        sho, pent = shiftdct[(i, at)]
        diff = cmpdct[at].get(i, 0.0)
        records.append((i + 1, at, sho, pent, diff))
    return records


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute(obs_shifts: dict, pred_shifts: dict, seq: str,
            min_aic: float = 5.0) -> ChezodResult:
    """Compute CheZOD Z-scores and CheSPI principal components.

    Parameters
    ----------
    obs_shifts  : from io.get_chemical_shifts()
    pred_shifts : from potenci.predict()
    seq         : one-letter amino acid sequence string
    min_aic     : AIC threshold for automatic re-referencing

    Returns
    -------
    ChezodResult
    """
    # 1. Compute secondary shifts (obs - pred)
    cmpdct, shiftdct = _cmp2pred(obs_shifts, pred_shifts, seq)

    if not cmpdct:
        raise ValueError("No matching chemical shifts found between observed and predicted data")

    total_bb_shifts = sum(len(cmpdct[at]) for at in cmpdct)
    print(f"Total backbone shifts for CheZOD: {total_bb_shifts}")

    # 2. Estimate running offset
    offr = _compute_zscores_and_pcs(cmpdct, shiftdct, seq,
                                     offdct=None, min_aic=min_aic)

    if offr is not None:
        print("Running offset estimated, evaluating with and without correction...")
        # Zero-correction dict
        off0 = {at: 0.0 for at in offr}
        # Evaluate corrected
        result_cor = _compute_zscores_and_pcs(cmpdct, shiftdct, seq,
                                               offdct=offr, min_aic=min_aic,
                                               return_offsets=False)
        armsdc, avc_cor, cdfs3c = result_cor
        # Evaluate uncorrected
        result_unc = _compute_zscores_and_pcs(cmpdct, shiftdct, seq,
                                               offdct=off0, min_aic=min_aic,
                                               return_offsets=False)
        armsd0, av0, cdfs30 = result_unc

        # Decide which to use
        use_uncorrected = av0 < avc_cor
        print(f"  av_uncorrected={av0:.3f} av_corrected={avc_cor:.3f} "
              f"→ {'no correction' if use_uncorrected else 'using correction'}")

        if use_uncorrected:
            # Get final offsets with no correction
            armsd_f, frac_f, noff_f, _ = _compute_zscores_and_pcs(
                cmpdct, shiftdct, seq, offdct=off0, min_aic=min_aic,
                return_offsets=True)
        else:
            armsd_f, frac_f, noff_f, _ = _compute_zscores_and_pcs(
                cmpdct, shiftdct, seq, offdct=offr, min_aic=min_aic,
                return_offsets=True)
    else:
        print("Warning: could not estimate running offset; using zero correction")
        off0 = {at: 0.0 for at in _BBATNS}
        armsd_f, frac_f, noff_f, _ = _compute_zscores_and_pcs(
            cmpdct, shiftdct, seq, offdct=off0, min_aic=min_aic,
            return_offsets=True)

    # 3. Final computation with accepted offsets
    resids, cdfs3, pc1ws, pc2ws, pc3ws = _compute_zscores_and_pcs(
        cmpdct, shiftdct, seq, offdct=noff_f, min_aic=min_aic,
        return_components=True)

    # 4. Extract shift records for output writing
    shift_records = _extract_shift_records(cmpdct, shiftdct)

    # Determine mini (0-based index of first residue position)
    mini = min(min(cmpdct[at].keys()) for at in cmpdct)

    return ChezodResult(
        seq=seq,
        mini=mini,
        residues=resids,
        zscores=array(cdfs3),
        pc1=array(pc1ws),
        pc2=array(pc2ws),
        pc3=array(pc3ws),
        shift_records=shift_records,
    )
