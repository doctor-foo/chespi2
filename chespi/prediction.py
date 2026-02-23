"""
prediction.py — 8-state / 3-state secondary structure prediction.

Uses a genetic algorithm (SSopt / SOPopulation) to fit CORVALS6 PC
correlations to the observed CheSPI principal components, producing
per-residue SS probability distributions.

Public interface:
    predict(result, first_resnum=1) -> SSResult
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from random import choice as randchoice, normalvariate, randint, uniform

import numpy as np
from numpy import zeros, array, exp, log, cumsum, hstack

from chespi.chezod import ChezodResult
from chespi.ga import GenericIndividual, Population

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent / "data"

# Ordered 1-letter codes matching alphabetical 3-letter sort
# ALA ARG ASP ASN CYS GLU GLN GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL
AA1S3 = list("ARDNCEEQGHILKMFPSTWYV")
AA1S3 = ['A','R','D','N','C','E','Q','G','H','I','L','K','M','F','P','S','T','W','Y','V']

ALLSS8  = 'HGIE-TSB'
SS8TO3  = {'H':'H','G':'H','I':'H','E':'S','-':'C','T':'C','S':'C','B':'C'}
SS3S    = 'HSC'
INDSS3  = {'H':[0,1,2], 'S':[3], 'C':[4,5,6,7]}

# ssigs: per-8-state-SS sigma for PC1 and PC2  (shape 2×8)
_SSIGS_RAW = (2.110,3.071,0.0034, 2.858,3.947,0.0219,
              3.897,3.337,0.1231, 3.663,3.490,-0.3100,
              3.116,3.344,-0.2312,2.980,3.670,-0.1521,
              2.932,3.532,-0.1583,3.876,3.886,-0.3073)
SSIGS = array(_SSIGS_RAW).reshape(8, 3).T[:2, :] * 1.1   # shape (2, 8)

# Prior fractional occupancy per amino acid for 8 SS states (HGIE-TSB order)
# Rows indexed by 'GPCTNSVWFHDYIMLKRQAE'
_FMAT_AAS = 'GPCTNSVWFHDYIMLKRQAE'
_FMAT = [
    [0.14211,0.03448,0.00052,0.14263,0.20585,0.27795,0.18652,0.00993],
    [0.15732,0.05774,0.00000,0.09958,0.40084,0.17908,0.09289,0.01255],
    [0.21405,0.04682,0.00000,0.32441,0.23077,0.09699,0.07358,0.01338],
    [0.23194,0.03681,0.00069,0.29375,0.22986,0.10208,0.08958,0.01528],
    [0.22014,0.04991,0.00000,0.14973,0.25312,0.19608,0.11854,0.01248],
    [0.24804,0.06414,0.00000,0.21270,0.23495,0.12173,0.10471,0.01374],
    [0.27904,0.01946,0.00000,0.44024,0.14508,0.05447,0.04780,0.01390],
    [0.36533,0.06667,0.00000,0.26933,0.13600,0.08800,0.05333,0.02133],
    [0.33894,0.04248,0.00000,0.30442,0.16372,0.07080,0.06814,0.01150],
    [0.30162,0.05206,0.00000,0.20826,0.20287,0.13824,0.07899,0.01795],
    [0.27330,0.06214,0.00063,0.11921,0.25618,0.15916,0.11921,0.01015],
    [0.33054,0.03975,0.00105,0.30439,0.14644,0.09205,0.06485,0.02092],
    [0.32101,0.02190,0.00000,0.38672,0.15674,0.04517,0.05270,0.01574],
    [0.42247,0.02472,0.00000,0.26292,0.16629,0.05393,0.06067,0.00899],
    [0.41264,0.03732,0.00000,0.26463,0.15098,0.06616,0.05683,0.01145],
    [0.38018,0.04684,0.00000,0.17108,0.15954,0.14460,0.08622,0.01154],
    [0.38118,0.04386,0.00000,0.21850,0.16268,0.09888,0.08134,0.01356],
    [0.44944,0.04831,0.00000,0.15843,0.16180,0.09438,0.07528,0.01236],
    [0.47164,0.04645,0.00000,0.18669,0.13444,0.09335,0.06163,0.00581],
    [0.44922,0.05857,0.00062,0.15016,0.14081,0.11776,0.07726,0.00561],
]


# ---------------------------------------------------------------------------
# Load CORVALS6
# ---------------------------------------------------------------------------

def _load_corvals() -> dict:
    path = _DATA_DIR / "corvals6.json"
    raw = json.loads(path.read_text())
    # Stored as [[list, list, dict]] per key; convert back to tuple
    return {k: (v[0], v[1], v[2]) for k, v in raw.items()}


CORVALS6 = _load_corvals()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _choose_random_consecutive(numelem: int, p: float = 0.2) -> list:
    first = np.random.randint(0, 2)
    lst = []
    for _ in range(numelem):
        if np.random.rand() < p:
            first = 1 - first
        lst.append(first)
    return lst


def _shannon(p: np.ndarray) -> float:
    z = p == 0
    c = p * np.log(p)
    c[z] = 0.0
    return float(np.sum(c))


def _selector(probs) -> int:
    prob = np.random.rand()
    cum = 0.0
    for j, probj in enumerate(probs):
        cum += probj
        if prob < cum:
            return j
    return len(probs) - 1


def _fastselector(probs, makecum: bool = True) -> int:
    if makecum:
        probs = cumsum(probs)
    gt = probs > np.random.rand()
    try:
        return list(gt).index(True) - 1
    except ValueError:
        return -1


def _get_score(pr: float, p0: float) -> tuple[float, str]:
    if pr >= 1.0:
        score = 1 + p0 / 0.4
        lab = '9' if score > 1.1 else '8'
    else:
        score = pr
        if pr > 0.96:   lab = '7'
        elif pr > 0.90: lab = '6'
        elif pr < 0.60: lab = '1'
        else:           lab = str(max(0, int(pr * 10) - 3))
    return score, lab


# ---------------------------------------------------------------------------
# Parameter initialisation
# ---------------------------------------------------------------------------

def _init_corvals() -> list[dict]:
    """Build per-PC parameter dicts from CORVALS6."""
    params: list[dict] = [{}, {}, {}]
    gnums = (0, 3, 4)   # canonical SS8 index for H, S, C

    Hs = ([0.8896, 0.1104, 0.0001], 'HGI')
    Cs = ([0.5621, 0.2402, 0.1736, 0.0241], '-TSB')
    ssprobs = {'H': Hs, 'C': Cs}

    for sellab, (aar, ssr, dct8) in CORVALS6.items():
        ss    = sellab[0]
        pcnum = int(sellab[1]) - 1
        A = array(aar).reshape((9, 20))
        N = array(ssr).reshape((2, 4, 8))
        if 'D0' not in params[pcnum]:
            params[pcnum]['D0'] = dict(dct8)
        else:
            params[pcnum]['D0'].update(dct8)
        params[pcnum][ss] = A, N

    for pcnum in range(3):
        params[pcnum]['NS'] = []
        for i, ss in enumerate(SS3S):
            A, N = params[pcnum][ss]
            refval = params[pcnum]['D0'][ALLSS8[gnums[i]]]
            nsum = np.sum(np.sum(N, axis=1), axis=0)
            if ss == 'S':
                gsnsum = nsum[gnums[i]]
            else:
                probs_w, _ = ssprobs[ss]
                gsnsum = float(np.average(
                    [nsum[gnums[i] + k] for k in range(len(probs_w))],
                    weights=probs_w))
            params[pcnum]['NS'].append(gsnsum)

    return params


def _update_params_with_seq(params: list[dict], seq: list[str]) -> None:
    """Fill S8 matrix (per-residue, per-8-state predicted PC) into params."""
    indss3 = {'H': [0,1,2], 'S': [3], 'C': [4,5,6,7]}
    numres = len(seq)
    seq4 = seq + ['G', 'G', 'G', 'G']

    for pcnum in range(3):
        S = zeros((3, numres))
        S8 = zeros((8, numres))
        for i, ss in enumerate(SS3S):
            A, N = params[pcnum][ss]
            for n in range(numres):
                cgn = sum(A[k + 4][AA1S3.index(seq4[n + k])] for k in range(-4, 5))
                S[i, n] = cgn
                for j in indss3[ss]:
                    S8[j, n] = cgn + params[pcnum]['D0'][ALLSS8[j]]
        params[pcnum]['S8'] = S8


# ---------------------------------------------------------------------------
# Bayesian initial SS guess
# ---------------------------------------------------------------------------

def _guesss8s(params, resis, pcsobs, priors, ssigs, s8obs=None):
    """Compute posterior SS probabilities using observed PCs and priors."""
    ru = array(resis - 1, dtype=int)
    ss8to3ind = {ss: i for i, ss_group in enumerate(['HGI', 'E', '-TSB'])
                 for ss in ss_group}
    inds3 = array([0,0,0,1,2,2,2,2])

    sig = array([[[ssigs[pcnum, i] for _ in range(len(ru))] for i in range(8)]
                 for pcnum in (0, 1)])
    preds = [
        array([params[pc]['S8'][i] + params[pc]['NS'][inds3[i]] for i in range(8)])
        for pc in (0, 1)
    ]
    preds = [prp[:, ru] for prp in preds]
    dev = array([preds[pc] - pcsobs[pc] for pc in (0, 1)])
    dev /= sig
    probs = exp(-0.5 * dev ** 2) / sig
    probs0 = probs[0] * probs[1]
    priors0 = priors[:, ru]
    post0 = priors0 * probs0
    post = post0 / np.sum(post0, axis=0)

    newpri = priors.copy()
    newpri[:, ru] = post
    return newpri, post0


def _getmaxss(pri) -> tuple[list, list]:
    ss8ind = [pri[:, n].argmax() for n in range(pri.shape[1])]
    ss8 = [ALLSS8[i] for i in ss8ind]
    ss3 = [SS3S[{'H':0,'G':0,'I':0,'E':1,'-':2,'T':2,'S':2,'B':2}[s]] for s in ss8]
    return ss8, ss3


def _avenscorr(params, probs, numres):
    """Probability-weighted neighbour-SS correction sum."""
    X = array([
        [
            [
                [
                    [params[pc][ss][1][direc, k, j] * probs[j, n]
                     for j in range(8)]
                    for n in range(numres)
                ]
                for k in range(4)
            ]
            for direc in (0, 1)
        ]
        for ss in 'HSC'
        for pc in (0, 1)
    ])
    return np.sum(X, axis=(2, 3, 5))   # shape (2, 3)


# ---------------------------------------------------------------------------
# Segment bookkeeping
# ---------------------------------------------------------------------------

class Segments8(dict):
    """Segment dictionary over 8-state labels (read-only helper)."""

    def __init__(self, s8, allss='HGSTB-E'):
        super().__init__()
        self.s8 = s8
        self.s3 = s8
        p3 = None
        previ = 0
        sl = 1
        self.segm: list = []
        self._ar = zeros(len(s8), dtype=int)
        for i, s3i in enumerate(s8):
            if s3i == p3:
                sl += 1
            elif p3 is not None:
                self.setdefault(p3, {})[previ] = sl
                self._ar[previ:previ + sl] = len(self.segm)
                self.segm.append([p3, previ])
                sl = 1
                previ = i
            p3 = s3i
        self.setdefault(p3, {})[previ] = sl
        self._ar[previ:previ + sl] = len(self.segm)
        self.segm.append([s3i, previ])


class Segments(dict):
    """Mutable segment dictionary over 3-state labels (H/S/C)."""

    def __init__(self, s8=None, s3=None):
        super().__init__()
        if s8 is None:
            return
        self.s8 = s8
        self.s3 = s3
        self['H'] = {}
        self['S'] = {}
        self['C'] = {}
        p3 = None
        previ = 0
        sl = 1
        self.segm: list = []
        self._ar = zeros(len(s3), dtype=int)
        for i, s3i in enumerate(s3):
            if s3i == p3:
                sl += 1
            elif p3 is not None:
                self[p3][previ] = sl
                self._ar[previ:previ + sl] = len(self.segm)
                self.segm.append([p3, previ])
                sl = 1
                previ = i
            p3 = s3i
        self[p3][previ] = sl
        self._ar[previ:previ + sl] = len(self.segm)
        self.segm.append([p3, previ])

    # -- clone --

    def get_clone(self) -> "Segments":
        new = Segments()
        for ss in 'HSC':
            new[ss] = self[ss].copy()
        new.s8 = self.s8[:]
        new.s3 = self.s3[:]
        new.segm = [lst[:] for lst in self.segm]
        new._ar = self._ar.copy()
        return new

    # -- segment modification primitives --

    def delete_segment(self, i, ss, target):
        si = int(self._ar[i])
        s0c, i0c = self.segm[si]
        sl = self[s0c][i0c]
        for k in range(sl):
            self.s3[i + k] = target
        s0p = None
        s0s = 'none'
        if si > 0:
            s0p, i0p = self.segm[si - 1]
            slp = self[s0p][i0p]
        if len(self.segm) > si + 1:
            s0s, i0s = self.segm[si + 1]
            sls = self[s0s][i0s]
        if s0p == s0s:
            if s0p != target:
                self.modify_segment(i, ss, target)
                for k in range(sl - 1):
                    self.increment_segment(i, target, 1)
            else:
                self[ss].pop(i)
                self.segm.pop(si)
                self.segm.pop(si)
                self[s0s].pop(i0s)
                self[s0p][i0p] += sl + sls
                self._ar[i:i + sl] -= 1
                self._ar[i + sl:] -= 2
        elif s0p == target:
            self[ss].pop(i)
            self.segm.pop(si)
            self[s0p][i0p] += sl
            self._ar[i:] -= 1
        elif s0s == target:
            self[ss].pop(i)
            self.segm.pop(si)
            self[s0s].pop(i0s)
            self[s0s][i] = sls + sl
            self.segm[si][1] -= sl
            self._ar[i + sl:] -= 1
        else:
            self[ss].pop(i)
            self[target][i] = sl
            self.segm[si][0] = target

    def increment_segment(self, i, ss, delta):
        si = int(self._ar[i])
        s0c, i0c = self.segm[si]
        sl = self[s0c][i0c]
        if delta > 0:
            s0s, i0s = self.segm[si + 1]
            sls = self[s0s][i0s]
            if sls <= delta:
                self.delete_segment(i0s, s0s, s0c)
            else:
                self.s3[i0s:i0s + delta] = ss
                self[ss][i0c] += delta
                self[s0s].pop(i0s)
                self[s0s][i0s + delta] = sls - delta
                self.segm[si + 1][1] += delta
                self._ar[i0s:i0s + delta] -= 1
        elif delta < 0:
            s0p, i0p = self.segm[si - 1]
            slp = self[s0p][i0p]
            if slp <= abs(delta):
                self.delete_segment(i0p, s0p, s0c)
            else:
                self.s3[i0c + delta:i0c] = ss
                self[ss].pop(i0c)
                self[ss][i0c + delta] = sl - delta
                self[s0p][i0p] = slp + delta
                self.segm[si][1] += delta
                self._ar[i0c + delta:i0c] += 1

    def _decrement_segment_end(self, i, ss, si):
        self.s3[-1] = 'C'
        self[ss][i] -= 1
        end = len(self._ar) - 1
        self['C'][end] = 1
        self.segm.append(['C', end])
        self._ar[-1] += 1

    def _decrement_segment_start(self, i, ss):
        self.s3[0] = 'C'
        sl = self[ss].pop(0)
        self[ss][i + 1] = sl - 1
        self['C'][0] = 1
        self.segm.insert(0, ['C', 0])
        self.segm[1][1] = 1
        self._ar[1:] += 1

    def decrement_segment(self, i, ss, delta):
        si = int(self._ar[i])
        s0c, i0c = self.segm[si]
        if delta < 0:
            if si > 0:
                s0p, i0p = self.segm[si - 1]
                self.increment_segment(i0p, s0p, -delta)
            else:
                self._decrement_segment_start(i, s0c)
        elif delta > 0:
            if len(self.segm) > si + 1:
                s0s, i0s = self.segm[si + 1]
                self.increment_segment(i0s, s0s, -delta)
            else:
                self._decrement_segment_end(i, s0c, si)

    def modify_segment(self, i, ss, target):
        si = int(self._ar[i])
        s0c, i0c = self.segm[si]
        sl = self[s0c][i0c]
        self.s3[i] = target
        if i0c < i < i0c + sl - 1:
            self[s0c][i0c] = i - i0c
            self[target][i] = 1
            self[s0c][i + 1] = i0c + sl - i - 1
            self.segm.insert(si + 1, [target, i])
            self.segm.insert(si + 2, [s0c, i + 1])
            self._ar[i] += 1
            self._ar[i + 1:] += 2
        elif i == i0c:
            self[s0c].pop(i)
            self[target][i] = 1
            self.segm.insert(si + 1, [target, i])
            if sl > 1:
                self[s0c][i + 1] = sl - 1
                self.segm.insert(si + 2, [s0c, i + 1])
                self._ar[i + 1:] += 1
            self.segm.pop(si)
        elif i == i0c + sl - 1:
            self[s0c][i0c] = sl - 1
            self[target][i] = 1
            self.segm.insert(si + 1, [target, i])
            self._ar[i] += 1
            self._ar[i + 1:] += 1

    # -- remediation of disallowed sequences --

    def return_disallowed(self):
        for i in self['H']:
            if self['H'][i] < 3: return 'H', i, self['H'][i]
        for i in self['S']:
            if self['S'][i] < 2: return 'S', i, 1
        for i in self['H']:
            if self['H'][i] < 4:
                for k in range(3):
                    if self.s8[i + k] != 'G': return 'G', i, 3
        for i in self['H']:
            sl = self['H'][i]
            segstr = self.s8[i:i + sl]
            if 'G' in segstr:
                gi = segstr.index('G')
                if gi + 1 >= sl:          return 'g', i + gi, 1
                if segstr[gi + 1] != 'G': return 'g', i + gi, 1
                if gi + 2 >= sl:          return 'g', i + gi, 2
                if segstr[gi + 2] != 'G': return 'g', i + gi, 2
        return None

    def remedy_disallowed(self, stoch=None, subs=None):
        if stoch is None:
            stoch = {('H',1):(0.6,0.4), ('H',2):(0.1,0.9), ('S',1):(0.6,0.4), ('G',3):(0.0,1.0)}
        if subs is None:
            subs = {('H',1):(0.3,0.45,0.2,0.05), ('H',2):(0.2,0.65,0.1,0.05), ('S',1):(0.4,0.15,0.2,0.25)}
        coils = '-TSB'
        changes3, changes8 = [], []
        while True:
            dis = self.return_disallowed()
            if dis is None:
                break
            ss, i, sl = dis
            if ss in 'HS':
                pdel, pext = stoch[(ss, sl)]
                if np.random.rand() < pdel:
                    self.delete_segment(i, ss, 'C')
                    for k in range(sl):
                        changes3.append((i + k, 'C'))
                        ind = _selector(subs[(ss, sl)])
                        self.s8[i + k] = coils[ind]
                        changes8.append((i + k, coils[ind]))
                else:
                    ssp = self.s3[i - 1] if i > 0 else None
                    try:    sss = self.s3[i + sl]
                    except IndexError: sss = None
                    if ssp == 'C' and sss != 'C':   delta = -1
                    elif sss == 'C' and ssp != 'C': delta = 1
                    elif sss is None: delta = -1
                    elif i == 0:      delta = 1
                    elif ssp is None: delta = 1
                    else: delta = 1 - 2 * int(np.random.rand() * 2)
                    self.increment_segment(i, ss, delta)
                    s8id = {'H':'H', 'S':'E'}[ss]
                    pos = i + delta if delta < 0 else i + delta + sl - 1
                    self.s8[pos] = s8id
                    changes8.append((pos, s8id))
            elif ss == 'G':
                for k in range(3):
                    self.s8[i + k] = 'G'
                    changes8.append((i + k, 'G'))
            elif ss == 'g':
                for k in range(sl):
                    self.s8[i + k] = 'H'
                    changes8.append((i + k, 'H'))
        return changes3, changes8

    # -- mutation helpers --

    def _modifys8(self, probs, i, changes8):
        coils = '-TSB'
        s8i = self.s8[i]
        s8val = s8i
        while s8val == s8i:
            ind = _selector(probs)
            s8val = coils[ind]
        self.s8[i] = s8val
        changes8.append((i, s8val))

    def choose_coil_point(self):
        coils_list = list(self['C'].items())
        if not coils_list: return None, None
        i0, sl = coils_list[randint(0, len(coils_list) - 1)]
        j = randint(0, sl - 1)
        i = i0 + j
        s8i = self.s8[i]
        probs = (0.4, 0.2, 0.3, 0.1)
        if (i > 0 and self.s8[i-1] == 'E') or (i < len(self.s8)-1 and self.s8[i+1] == 'E'):
            probs = (0.4, 0.3, 0.3, 0.0)
        coilvals = '-TSB'
        s8val = s8i
        while s8val == s8i:
            s8val = coilvals[_selector(probs)]
        return i, s8val

    def choose_incr_point(self):
        hs = list(self['H'].items())
        es = list(self['S'].items())
        if not hs and not es: return None, None, None
        ssm = ('S' if not hs else 'H') if (not es or not hs) else randchoice('HS')
        elems = list(self[ssm].items())
        i0, sl = elems[randint(0, len(elems) - 1)]
        if i0 == 0: direc = 'R'
        elif int(self._ar[i0]) == len(self.segm) - 1: direc = 'L'
        else: direc = randchoice('LR')
        if direc == 'R':
            i = i0 + sl; refi = i0 + sl - 1
        else:
            i = i0 - 1;  refi = i0
        return i, self.s8[refi], (i0, ssm, direc)

    def choose_delete_elem(self):
        hs = list(self['H'].items())
        es = list(self['S'].items())
        if not hs and not es: return None, None, None
        ssm = ('S' if not hs else 'H') if (not es or not hs) else randchoice('HS')
        elems = list(self[ssm].items())
        minsl = {'S': 2, 'H': 4}
        minelem = min(elems, key=lambda x: x[1])
        if minelem[1] > minsl[ssm] + 2: return None, None, None
        i0, sl = minelem
        subs = {'H': (0.5, 0.27, 0.2, 0.03), 'S': (0.5, 0.15, 0.2, 0.15)}
        coils = '-TSB'
        target = ''
        for _ in range(sl):
            target += coils[_selector(subs[ssm])]
        return i0, target, ssm

    def choose_decr_point(self, allowsmall=False):
        hs = list(self['H'].items())
        es = list(self['S'].items())
        if not hs and not es: return None, None, None
        ssm = ('S' if not hs else 'H') if (not es or not hs) else randchoice('HS')
        elems = list(self[ssm].items())
        minsl = {'S': 2, 'H': 4}
        maxelem = max(elems, key=lambda x: x[1])
        if maxelem[1] < minsl[ssm] and not allowsmall: return None, None, None
        i0, sl = 0, 0
        while sl < minsl[ssm]:
            i0, sl = elems[randint(0, len(elems) - 1)]
            if allowsmall: break
        direc = randchoice('LR')
        if direc == 'R': i = i0 + sl - 1; bi = i + 1
        else:            i = i0;           bi = i - 1
        if bi < 0 or bi == len(self.s3): bases3 = 'C'
        else: bases3 = self.s3[bi]
        coils = '-TSB'
        subs = {'H': (0.5, 0.27, 0.2, 0.03), 'S': (0.65, 0.15, 0.2, 0.0)}
        if bases3 == 'C':
            targets8 = coils[_selector(subs[ssm])]
        else:
            targets8 = {'H':'H', 'S':'E'}[bases3]
        return i, targets8, (i0, ssm, direc)

    def choose_split_point(self):
        hs = list(self['H'].items())
        es = list(self['S'].items())
        minsl = {'S': 2, 'H': 3}
        if not hs and not es: return None, None, None
        if not hs: ssm = 'S'
        elif not es: ssm = 'H'
        else:
            smaxls = max(e[1] - minsl['S'] * 2 for e in es)
            hmaxls = max(e[1] - minsl['H'] * 2 for e in hs)
            if smaxls < 1 and hmaxls < 1: return None, None, None
            ssm = ('S' if hmaxls < 1 else 'H') if smaxls > 0 and hmaxls < 1 else \
                  ('H' if smaxls < 1 else randchoice('HS'))
        elems = list(self[ssm].items())
        i0, sl = 0, 0
        while sl < minsl[ssm] * 2 + 1:
            i0, sl = elems[randint(0, len(elems) - 1)]
        j = randint(minsl[ssm], sl - minsl[ssm] - 1)
        i = i0 + j
        subs = {'H': (0.27, 0.5, 0.2, 0.03), 'S': (0.65, 0.15, 0.2, 0.0)}
        return i, '-TSB'[_selector(subs[ssm])], ssm

    def choose_overwriteH2G(self):
        hs = list(self['H'].items())
        if not hs: return None, None, None
        elems = list(self['H'].items())
        minls = min(e[1] for e in elems)
        if minls > 4:
            i0, sl = elems[randint(0, len(elems) - 1)]
            direc = randchoice('LR')
            if direc == 'L':
                i = i0
                gsl = [3, min(sl, 4), min(sl, 5)][_selector((0.6, 0.3, 0.1))]
                if 'H' not in self.s8[i:i + sl]: return None, None, None
            else:
                gsl = [3, min(sl, 4), min(sl, 5)][_selector((0.6, 0.3, 0.1))]
                if 'H' not in self.s8[i0 + sl - gsl:i0 + sl]: return None, None, None
                i = i0 + sl - gsl
        else:
            direc = 'a'
            gsl = 999
            while gsl > 4:
                i, gsl = elems[randint(0, len(elems) - 1)]
            if 'H' not in self.s8[i:i + gsl]: return None, None, None
        return i, 'G' * gsl, (gsl, direc)

    def choose_overwriteC2G(self):
        cs = list(self['C'].items())
        if not cs: return None, None, None
        elems = list(self['C'].items())
        if min(e[1] for e in elems) < 3: return None, None, None
        i0, sl = 0, 0
        while sl < 3:
            i0, sl = elems[randint(0, len(elems) - 1)]
        j = randint(0, sl - 3)
        return i0 + j, 'GGG', sl

    def get_s8_mutation(self, flag, ind, i, ss8) -> list:
        s8new = self.s8[:]
        if flag in ['coil', 'incr', 'decr', 'split']:
            s8new[i] = ss8
        else:
            for k, s8k in enumerate(ss8):
                s8new[i + k] = s8k
        return s8new

    def execute_mutation(self, flag, ind, i, ss8, info):
        if flag == 'coil':
            self.s8[i] = ss8
        elif flag == 'incr':
            self.s8[i] = ss8
            i0, ssm, direc = info
            self.increment_segment(i0, ssm, 1 if direc == 'R' else -1)
        elif flag == 'decr':
            self.s8[i] = ss8
            i0, ssm, direc = info
            self.decrement_segment(i0, ssm, 1 if direc == 'R' else -1)
        elif flag == 'split':
            self.s8[i] = ss8
            self.modify_segment(i, info, 'C')
        elif flag == 'del':
            for k, s8k in enumerate(ss8): self.s8[i + k] = s8k
            self.delete_segment(i, info, 'C')
        elif flag == 'H2G':
            for k, s8k in enumerate(ss8): self.s8[i + k] = s8k
        elif flag == 'C2G':
            for k, s8k in enumerate(ss8): self.s8[i + k] = s8k
            self.modify_segment(i, 'C', 'H')
            for k in (1, 2): self.increment_segment(i, 'H', 1)


# ---------------------------------------------------------------------------
# SSparameters — parameter container (environment for the GA)
# ---------------------------------------------------------------------------

class SSparameters:
    """Parameter container: priors, CORVALS6 params, observed PCs."""

    allss8 = ALLSS8
    ssigs  = SSIGS

    def __init__(self, entry_id: str = ''):
        self.entry_id = entry_id

    def init_priors_basic(self) -> np.ndarray:
        """Build (8 × N) prior matrix from per-residue amino-acid statistics."""
        pri = []
        for aa in self.seq:
            if aa in _FMAT_AAS:
                fracs = _FMAT[_FMAT_AAS.index(aa)]
            else:
                fracs = [1/8] * 8
            pri.append(array(fracs))
        return array(pri).T   # shape (8, N)

    def set_input(self, seq, resis, pc1sobs, pc2sobs, zsco):
        self.seq = list(seq)
        self.resis = array(resis)
        self.mini  = int(self.resis[0])
        self.maxi  = int(self.resis[-1])
        self.ru    = array(self.resis - 1, dtype=int)
        self.pcsobsref = (pc1sobs, pc2sobs)
        self.zscores   = array(zsco)
        self.disordered = self.zscores < 8.0
        self.s8obs = None
        self.s3obs = None

    def initparameters(self):
        self.priors = self.init_priors_basic()
        # Force disordered-region priors toward coil / bend
        for i, ires in enumerate(self.resis):
            if self.disordered[i]:
                self.priors[:, int(ires) - 1] = [0, 0, 0, 0.05, 0.8, 0, 0.2, 0]
        self.params = _init_corvals()
        _update_params_with_seq(self.params, self.seq)

    def make_guess(self):
        self.ss8priors, _ = _guesss8s(
            self.params, self.resis, self.pcsobsref,
            self.priors, self.ssigs)


# ---------------------------------------------------------------------------
# SSopt — genetic algorithm individual
# ---------------------------------------------------------------------------

class SSopt(GenericIndividual):

    _flags = ['coil', 'incr', 'decr', 'split', 'del', 'H2G', 'C2G']

    def __init__(self, ssp: SSparameters, s8: list, s3: list | None = None):
        self.ssp = ssp
        self.s8  = s8
        if s3 is None:
            s3 = [SS3S[{'H':0,'G':0,'I':0,'E':1,'-':2,'T':2,'S':2,'B':2}[x]] for x in s8]
        self.s3 = s3
        self.ss8inds = [ALLSS8.index(x) for x in (self.s8 + ['-', '-', '-', '-'])]

    # -- clone --

    def get_clone(self, need_segments: bool = False) -> "SSopt":
        new = SSopt(self.ssp, self.s8[:], self.s3[:])
        if need_segments:
            new.segments = self.segments.get_clone()
        new.S9s = [s9.copy() for s9 in self.S9s]
        new.post0ref = self.post0ref.copy()
        return new

    # -- PC back-calculation --

    def backcalcPCs(self, pcnum: int):
        ssp = self.ssp
        numres = len(ssp.seq)
        pari = ssp.params[pcnum]
        S8 = pari['S8']
        self.ss8inds = [ALLSS8.index(x) for x in (self.s8 + ['-', '-', '-', '-'])]
        ss8inds = self.ss8inds
        s8vals = [S8[ss8inds[n], n] for n in range(numres)]
        S9 = zeros((9, numres))
        S9[4] = s8vals
        for n in range(numres):
            gssn = self.s3[n]
            N = pari[gssn][1]
            S9[:4, n] = [N[0, k, ss8inds[n - 1 - k]] for k in range(3, -1, -1)]
            S9[5:,  n] = [N[1, k, ss8inds[n + 1 + k]] for k in range(4)]
        return np.sum(S9, axis=0), S9

    def backcalcbothPCs(self):
        self.backpcs = []
        self.S9s = []
        for pcnum in (0, 1):
            pcp, S9p = self.backcalcPCs(pcnum)
            self.backpcs.append(pcp)
            self.S9s.append(S9p)

    def backcalcmut(self, n0: int, ss8m: str):
        numres = len(self.ssp.seq)
        ss8inds = [ALLSS8.index(x) for x in (self.s8 + ['-', '-', '-', '-'])]
        s3new = self.s3[:]
        M = len(ss8m)
        for k in range(M):
            ss8ind = ALLSS8.index(ss8m[k])
            ss8inds[n0 + k] = ss8ind
            s3new[n0 + k] = SS8TO3[ss8m[k]]
        newS9s = [s9.copy() for s9 in self.S9s]
        for n in range(n0, n0 + M):
            ss8ind = ss8inds[n]
            for pcnum in (0, 1):
                pari = self.ssp.params[pcnum]
                S9 = newS9s[pcnum]
                S9[4, n] = pari['S8'][ss8ind, n]
                gssn = s3new[n]
                N = pari[gssn][1]
                S9[:4, n] = [N[0, k, ss8inds[n - 1 - k]] for k in range(3, -1, -1)]
                S9[5:,  n] = [N[1, k, ss8inds[n + 1 + k]] for k in range(4)]
                for k in range(4):
                    if n + 1 + k < numres:
                        S9[3 - k, n + 1 + k] = pari[s3new[n + 1 + k]][1][0, k, ss8ind]
                    if n - 1 - k >= 0:
                        S9[5 + k, n - 1 - k] = pari[s3new[n - 1 - k]][1][1, k, ss8ind]
        return newS9s, ss8inds

    # -- likelihood --

    def calcpostlik(self) -> float:
        ru = self.ssp.ru
        predsnew = [pcp[ru] for pcp in self.backpcs]
        sig = array([[self.ssp.ssigs[pcnum, ALLSS8.index(self.s8[int(n) - 1])]
                      for n in self.ssp.resis] for pcnum in (0, 1)])
        dev = array([predsnew[pc] - self.ssp.pcsobsref[pc] for pc in (0, 1)])
        dev /= sig
        probs0 = exp(-0.5 * dev ** 2)[0] / sig[0] * exp(-0.5 * dev ** 2)[1] / sig[1]
        priors = array([self.ssp.priors[self.ss8inds[int(n) - 1], n - 1]
                        for n in self.ssp.resis])
        post0ref = probs0 * priors
        self.post0ref = post0ref
        self.score = float(np.sum(log(post0ref)))
        self.energy = -self.score
        return self.score / len(post0ref)

    def calcpostlik_local(self, lr, nmut, ss8inds, s8, backpcs):
        ru = self.ssp.ru
        tru = (ru < lr[1]) & (ru >= lr[0])
        lru = ru[tru]
        if len(lru) < 1:
            return -999, 0
        minru = list(ru).index(min(lru))
        newru = lru - nmut + 4
        newru -= max(0, 4 - nmut)
        predsnew = [pcp[newru] for pcp in backpcs]
        pcsobs = [self.ssp.pcsobsref[pc][minru:minru + len(lru)] for pc in (0, 1)]
        sig = array([[self.ssp.ssigs[pcnum, ALLSS8.index(s8[n])] for n in newru]
                     for pcnum in (0, 1)])
        dev = array([predsnew[pc] - pcsobs[pc] for pc in (0, 1)])
        dev /= sig
        probs0 = exp(-0.5 * dev ** 2)[0] / sig[0] * exp(-0.5 * dev ** 2)[1] / sig[1]
        priors = array([self.ssp.priors[ss8inds[int(n) - 1], n - 1] for n in lru + 1])
        post0ref = probs0 * priors
        self.post0refnewdata = post0ref, (minru, minru + len(lru))
        return float(np.sum(log(post0ref))), float(np.sum(log(self.post0ref[minru:minru + len(lru)])))

    def get_diff_mutation(self, n: int, ss8m: str):
        newS9s, newss8inds = self.backcalcmut(n, ss8m)
        N = len(self.ssp.seq)
        M = len(ss8m)
        lr = (max(0, n - 4), min(n + 4 + M, N))
        backpcs = [np.sum(newS9s[pc][:, lr[0]:lr[1]], axis=0) for pc in (0, 1)]
        news8loc = self.s8[lr[0]:n] + list(ss8m) + self.s8[n + M:lr[1]]
        locpost, oldloc = self.calcpostlik_local(lr, n, newss8inds, news8loc, backpcs)
        self.newS9s = newS9s
        return locpost, locpost - oldloc

    # -- fitness --

    def calculate_fitness(self):
        self.backcalcbothPCs()
        self.calcpostlik()

    # -- mutation / crossover --

    def initialize_random(self) -> "SSopt":
        pri = self.ssp.ss8priors
        ss8ind = [_fastselector(hstack(([0], pri[:, n]))) for n in range(pri.shape[1])]
        ss8 = [ALLSS8[i] for i in ss8ind]
        ss3 = [SS3S[{'H':0,'G':0,'I':0,'E':1,'-':2,'T':2,'S':2,'B':2}[x]] for x in ss8]
        new = SSopt(self.ssp, ss8, ss3)
        new.backcalcbothPCs()
        new.calcpostlik()
        new.init_segments()
        new.segments.remedy_disallowed()
        return new

    def init_segments(self):
        self.segments = Segments(self.s8, self.s3)

    def choose_mutation(self):
        probs = (0.3, 0.2, 0.2, 0.05, 0.05, 0.12, 0.08)
        ind = _selector(probs)
        flag = self._flags[ind]
        if flag == 'coil':   i, ss8, info = *self.segments.choose_coil_point(), 'coil'
        elif flag == 'incr': i, ss8, info = self.segments.choose_incr_point()
        elif flag == 'decr': i, ss8, info = self.segments.choose_decr_point()
        elif flag == 'split':i, ss8, info = self.segments.choose_split_point()
        elif flag == 'del':  i, ss8, info = self.segments.choose_delete_elem()
        elif flag == 'H2G':  i, ss8, info = self.segments.choose_overwriteH2G()
        elif flag == 'C2G':  i, ss8, info = self.segments.choose_overwriteC2G()
        if i is None:
            return self.choose_mutation()
        return ind, i, ss8, info

    def crossover(self, other: "SSopt") -> "SSopt":
        M = len(self.s8)
        inds = _choose_random_consecutive(M, p=0.2)
        both = (self, other)
        newss8 = [both[inds[i]].s8[i] for i in range(M)]
        return SSopt(self.ssp, newss8)

    def multicrossover(self, popul, selrat: float, size: int) -> "SSopt":
        M = len(self.s8)
        nums = [popul.selectNormal(selrat, size) for _ in range(M)]
        newss8 = [popul[nums[i]].s8[i] for i in range(M)]
        return SSopt(self.ssp, newss8)

    def getid(self) -> str:
        return ''.join(self.s8)

    def get_class_stats(self, popul, statlen=None):
        M = len(self.s8)
        if statlen is None:
            statlen = len(popul)
        A = [[ssopt.s8[i] for ssopt in popul[:statlen]] for i in range(M)]
        C = array([[A[i].count(s) for s in ALLSS8] for i in range(M)], dtype=float) / statlen
        entrs = [_shannon(C[i]) for i in range(M)]
        spread = float(exp(-np.mean(entrs)))
        probs = C.T   # shape (8, M)
        return -9.0, spread, C   # trueprobs not computed (no observed)


# ---------------------------------------------------------------------------
# SOPopulation — GA population specialised for SS optimisation
# ---------------------------------------------------------------------------

class SOPopulation(Population):

    def derive_stats(self, cnt: int) -> None:
        energies = array([obj.energy for obj in self])
        _, genestd, _ = self[0].get_class_stats(self)
        print(
            f"  gen {cnt:4d} | E_min={energies.min():.3f} "
            f"E_avg={np.mean(energies):.3f} spread={genestd:.5f}  "
            f"{self[0].getid()[:60]}"
        )

    def collect_results(self):
        """Return per-residue prediction data (no file I/O)."""
        obj   = self[0]
        _, _, fracs = obj.get_class_stats(self)  # fracs shape (N, 8)
        ru    = list(obj.ssp.ru)

        probs8_list, probs3_list = [], []
        s8_list, s3_list = [], []
        prob8_list, prob3_list, post_list = [], [], []
        conf8, conf3 = '', ''

        for i in range(len(fracs)):
            pri  = fracs[i]                   # shape (8,)
            p0i  = float(obj.post0ref[ru.index(i)]) if i in ru else 0.0
            pri3 = array([sum(pri[j] for j in INDSS3[k]) for k in 'HSC'])

            mip  = int(pri.argmax())
            mip3 = int(pri3.argmax())

            s8max = obj.s8[i]
            s3max = obj.s3[i]

            _, c8 = _get_score(float(pri[mip]),  p0i)
            _, c3 = _get_score(float(pri3[mip3]), p0i)

            probs8_list.append(pri.tolist())
            probs3_list.append(pri3.tolist())
            s8_list.append(s8max)
            s3_list.append(s3max)
            prob8_list.append(float(pri[mip]))
            prob3_list.append(float(pri3[mip3]))
            post_list.append(p0i)
            conf8 += c8
            conf3 += c3

        return probs8_list, probs3_list, s8_list, s3_list, prob8_list, prob3_list, post_list, conf8, conf3

    def breed(self, limitfac=100.0, temperature=0.3, probs=(0.6, 0.3, 0.1),
              growthmode='replace', selrats=(0.5, 2.0), sortnum=10):
        cnt = 0
        numrep = 0
        size = len(self)
        flags = self._flags if hasattr(self, '_flags') else SSopt._flags
        dct: dict = {obj.getid(): obj for obj in self}
        self.iddct = dct

        while cnt < size * limitfac:
            i = self.selectNormal(selrats[0], size)
            obji = self[i]
            rn = uniform(0.0, 1.0)
            psum = 0.0
            j = 0
            for jj, pj in enumerate(probs):
                psum += pj
                if rn < psum:
                    j = jj
                    break

            prevener = obji.energy

            if j == 0:
                # mutation
                ind, I, ss8, info = obji.choose_mutation()
                locpost, locdiff = obji.get_diff_mutation(I, ss8)
                enerdiff = -locdiff
                childid = ''.join(obji.segments.get_s8_mutation(flags[ind], ind, I, ss8))
                repi = i
            elif j == 1:
                # crossover
                onum = i
                while onum == i:
                    onum = self.selectNormal(selrats[1], size)
                objo = self[onum]
                child = obji.crossover(objo)
                child.init_segments()
                child.segments.remedy_disallowed()
                repi = i if prevener >= objo.energy else onum
                childid = child.getid()
            else:
                # multicrossover
                onum = repi = -1
                child = self[-1].multicrossover(self, selrats[1], size)
                child.init_segments()
                child.segments.remedy_disallowed()
                childid = child.getid()

            if childid not in dct:
                if j > 0:
                    child.calculate_fitness()
                    enerdiff = child.energy - prevener
                ptest = 999
                if enerdiff > 0:
                    ptest = exp(-enerdiff / temperature)
                    if repi == 0:
                        ptest = -999
                if ptest > uniform(0.0, 1.0):
                    if j == 0:
                        child = obji.get_clone(need_segments=True)
                        child.segments.execute_mutation(flags[ind], ind, I, ss8, info)
                        child.S9s = obji.newS9s
                        post0refnew, ranges = obji.post0refnewdata
                        child.post0ref[ranges[0]:ranges[1]] = post0refnew
                        child.energy = prevener + enerdiff
                    dct[childid] = child
                    numrep += 1
                    if growthmode == 'replace':
                        self[repi] = child
                    else:
                        if j == 0:
                            self[repi] = child
                        else:
                            self.append(child)
                    if numrep % sortnum == 0:
                        self.sort('energy')
                        self.derive_stats(cnt)
            cnt += 1

    @property
    def _flags(self):
        return SSopt._flags


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class SSResult:
    seq:        list   # ['A','R',...] length N
    resi_list:  list   # [1, 2, ...] 1-indexed residue numbers
    probs8:     list   # list of N 8-element lists
    probs3:     list   # list of N 3-element lists
    s8_list:    list   # N 1-char strings (best 8-state)
    s3_list:    list   # N 1-char strings (best 3-state)
    prob8_list: list   # N floats (max 8-state prob)
    prob3_list: list   # N floats (max 3-state prob)
    post_list:  list   # N floats (posterior reference)
    conf8:      str    # N-char confidence string (8-state)
    conf3:      str    # N-char confidence string (3-state)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict(result: ChezodResult, first_resnum: int = 1) -> SSResult:
    """Run CheSPI 8-state / 3-state secondary structure prediction.

    Parameters
    ----------
    result       : ChezodResult from chespi.chezod.compute()
    first_resnum : residue number of result.seq[0] (default 1)

    Returns
    -------
    SSResult with per-residue probability distributions and best predictions.
    """
    seq   = list(result.seq)
    resis = array(result.residues)
    pc1s  = result.pc1
    pc2s  = result.pc2
    zsco  = result.zscores

    ssp = SSparameters()
    ssp.set_input(seq, resis, pc1s, pc2s, zsco)
    ssp.initparameters()
    ssp.make_guess()

    ss8max, ss3max = _getmaxss(ssp.ss8priors)
    ssopt = SSopt(ssp, ss8max, ss3max)
    ssopt.backcalcbothPCs()
    ssopt.init_segments()
    ssopt.segments.remedy_disallowed()
    ssopt.calcpostlik()

    popul = SOPopulation()
    popul.envi = ssp
    popul.fill_from_random(25, ssopt)
    popul.append(ssopt)
    popul.sort('energy')
    popul.breed(limitfac=75, growthmode='append')
    popul.cull(100)

    print(f"CheSPI GA done — best: {''.join(popul[0].s8)}")

    p8, p3, s8, s3, pb8, pb3, post, c8, c3 = popul.collect_results()
    N = len(seq)
    resi_list = [first_resnum + i for i in range(N)]

    return SSResult(
        seq=seq,
        resi_list=resi_list,
        probs8=p8,
        probs3=p3,
        s8_list=s8,
        s3_list=s3,
        prob8_list=pb8,
        prob3_list=pb3,
        post_list=post,
        conf8=c8,
        conf3=c3,
    )
