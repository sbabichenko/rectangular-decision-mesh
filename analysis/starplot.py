"""Star plots for the rectangular mesh: one coefficient's admission decision, re-litigated by eye.

The triangular repo's analysis/starplot.py with the rectangular geometry underneath. A page has
the plane (the cells the coefficient's column reaches, its pools coloured by the standardized
residual and sized by their share of the coefficient's information w_i x_i^2 / xTx), two
cross-sections through the vertex (along its parent segment and across it: observed log-odds by
bin against the fitted surface and the null surface with the surplus removed), and the
accounting card. The column is the hierarchical hat: the coefficient's influence carried to every
descendant at one half per generation and read bilinearly in the leaf cells, which is the column
the gate scored and the fit uses. Reads the run dumps (the binary's or the port's) and the design.

    python3 -m analysis.starplot <run_dir> <design.csv> --vertex 111 --out star.png
    python3 -m analysis.starplot <run_dir> <design.csv> --pick giant|keff|surplus
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd

from pyrect.estimator.design import K_ROOT_PRIOR_VARIANCE


def deviance(k, n, p) -> np.ndarray:
    """Binomial deviance per row at rates p against the saturated fit."""
    k = np.asarray(k, dtype=float); n = np.asarray(n, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        sat = np.where(k > 0, k * np.log(k / (n * p)), 0.0) + np.where(n - k > 0, (n - k) * np.log((n - k) / (n - n * p)), 0.0)
    return 2.0 * sat


def address_name(x: float, y: float) -> str:
    """The vertex's dyadic address: L the smallest level at which both coordinates are integers
    multiples of 2^-L, written "L5 (11, 22)" (the dumps print ten significant digits)."""
    for L in range(31):
        s = 2.0 ** L; tol = min(max(s * 2e-10, 1e-9), 1e-3)
        a, b = round(x * s), round(y * s)
        if abs(x * s - a) < tol and abs(y * s - b) < tol:
            return f"L{L} ({a}, {b})"
    return f"({x:.6g}, {y:.6g})"



class Run:
    def __init__(self, run_dir: Path, design: Path, law: Path | None = None, artifact_months: float = 0.0,
                 compare: Path | None = None, compare_label: str | None = None):
        # artifact_months > 0 hatches loan ages below it: the cumulative-attrition proxy (Ginnie) is
        # invalid there. A monthly snapshot such as SMM has no such region; leave it at 0.
        self.artifact_months = artifact_months
        self.dir = Path(run_dir)
        self.combined = None
        self.v = pd.read_csv(self.dir / "run_hier_vertices.csv").set_index("id", drop=False)
        self.cells = pd.read_csv(self.dir / "run_mesh.csv")
        if "h00" not in self.cells and "h000" not in self.cells:
            raise ValueError(f"{self.dir} is not a rectangular mesh run (run_mesh.csv has no cell corners)")
        # a three-dimensional run: the cells carry z bounds and eight corners (bit d of corner k: at hi[d])
        self.dim = 3 if "z0" in self.cells else 2
        self.CORNERS = tuple("".join("1" if (k >> d) & 1 else "0" for d in range(self.dim)) for k in range(1 << self.dim))
        self.lo_cols = ["x0", "y0", "z0"][:self.dim]; self.hi_cols = ["x1", "y1", "z1"][:self.dim]
        obs = pd.read_csv(self.dir / "run_obs.csv")
        d = pd.read_csv(design)
        obs["uses_mesh"] = True
        if self.combined and self.combined.get("msf_baseline") == "direct":
            if "group" not in d:
                raise ValueError("direct combined run is missing population labels")
            obs["uses_mesh"] = d.group.values != "M-SF"
        assert len(d) == len(obs), "design and run_obs row counts differ"
        if not np.array_equal(d.n0.values, obs.n.values) or not np.array_equal(d.k_term.values, obs.k.values):
            raise ValueError("design and run_obs outcomes are misaligned")
        model = self.dir / "run_model.json"
        self.model = json.loads(model.read_text()) if model.exists() else None
        self.height_scale = float(self.model.get("height_scale", 100.0)) if self.model else 100.0
        if self.model:   # the run's own covariate bounds and training-only baseline (dumps since 2026-09-16)
            self.wala_range = (self.model["wala_min"], self.model["wala_max"]); self.wac_range = (self.model["wac_min"], self.model["wac_max"])
        else:
            self.wala_range = (d.wala.min(), d.wala.max()); self.wac_range = (d.wac.min(), d.wac.max())
        obs["sx"] = (d.wala.values - self.wala_range[0]) / (self.wala_range[1] - self.wala_range[0])
        obs["sy"] = (d.wac.values - self.wac_range[0]) / (self.wac_range[1] - self.wac_range[0])
        obs["wala_m"] = d.wala.values; obs["wac_pct"] = d.wac.values
        self.coord_cols = ["sx", "sy"]
        if self.dim == 3:   # the third axis as the run placed it (log balance for the balance meshes)
            self.axis3 = self.model["axis3"]; self.axis3_range = (self.model["axis3_min"], self.model["axis3_max"])
            obs["sz"] = obs["z"].values if "z" in obs else (d[self.axis3].values - self.axis3_range[0]) / (self.axis3_range[1] - self.axis3_range[0])
            obs["bal"] = np.exp(self.axis3_range[0] + obs["sz"].values * (self.axis3_range[1] - self.axis3_range[0]))
            self.coord_cols.append("sz")
        obs["s2"] = d.sigma2_pool.values if "sigma2_pool" in d else (pd.read_csv(law).sigma2_pool.values if law else 0.0)
        # a design carrying sigma2_pool was fitted with the temper: the weight the fit gave pool i is I_i / (1 + I_i s2_i)
        if "multi_issuer" in d: obs["multi"] = d.multi_issuer.astype(bool).values; self.type_is_proxy = False
        elif "issue_type" in d: obs["multi"] = (d.issue_type.astype(str) == "M").values; self.type_is_proxy = False
        else: obs["multi"] = (d.n0.values >= 1000); self.type_is_proxy = True   # 2020 panel: 93-100% of pools >= 1000 loans are multi-issuer
        self.tempered = "sigma2_pool" in d
        # the pools that informed the surface: the training pools, unless the design says otherwise
        # (a combined-model run carries in_fit = 0 for the giants, which are scored but never fitted)
        obs["in_fit"] = d.in_fit.astype(bool).values if "in_fit" in d else (obs.is_train.values == 1)
        info = d.surface_info.values if "surface_info" in d else obs["info"].values
        obs["weight"] = info / (1 + info * obs["s2"]) if self.tempered else info
        self.obs = obs; self.design = d
        # another fit of the same design (row-aligned dumps), drawn beside this one in the sections
        self.compare = None
        if compare is not None:
            other = pd.read_csv(Path(compare) / "run_obs.csv")
            if len(other) != len(obs) or not np.array_equal(other.n.values, obs.n.values) or not np.array_equal(other.k.values, obs.k.values):
                raise ValueError("--compare run is not a fit of the same design")
            obs["p_other"] = other.p_hat.values
            self.compare = Path(compare)
        self.compare_label = compare_label or (self.compare.name if self.compare is not None else None)
        if self.model:
            self.pooled_logit = float(self.model["baseline_logit"])
        else:            # legacy dumps: recover the baseline from the fit
            self.pooled_logit = float(np.log(obs.p_hat / (1 - obs.p_hat)).sub(obs.g_hat / self.height_scale).median())
        # each pool's own base logit: the baseline plus its offset (levels, first-stage attributes) when the dump carries it
        obs["base"] = obs["offset"].values if "offset" in obs else self.pooled_logit
        # every candidate the gate scored, round by round (dumps since 2026-09-23)
        cand = self.dir / "run_candidates.csv"
        self.cand = pd.read_csv(cand) if cand.exists() else None
        # hanging vertices: on some cell's boundary but not its corner (an edge's interior in two
        # dimensions; in three the closure keeps them on edges too)
        c = self.cells; e = 1e-12
        vcols = ["x", "y", "z"][:self.dim]
        at_bound = np.ones((len(self.v), len(c)), bool); inside = np.ones_like(at_bound); on_face = np.zeros_like(at_bound); corner = np.ones_like(at_bound)
        for vc, lo, hi in zip(vcols, self.lo_cols, self.hi_cols):
            q = self.v[vc].values[:, None]; a_, b_ = c[lo].values[None, :], c[hi].values[None, :]
            bound = (np.abs(q - a_) < e) | (np.abs(q - b_) < e)
            inside &= (q > a_ - e) & (q < b_ + e); on_face |= bound; corner &= bound
        self.hanging = set(self.v.id.values[(inside & on_face & ~corner).any(axis=1)].tolist())
        # each vertex's children, for the influence pass
        self.children = {}
        for u in self.v.itertuples():
            for q in (int(u.parent0), int(u.parent1)):
                if q >= 0: self.children.setdefault(q, []).append(int(u.id))

    # ---- geometry ---------------------------------------------------------------------
    def coords(self, frame):
        """A pool frame's unit coordinates, one array per axis."""
        return [frame[c].values for c in self.coord_cols]

    def influence(self, vid: int) -> dict:
        """The coefficient's weight in every vertex's height: 1 at the vertex, and half of what its
        two parents carry at every descendant, free or not (the surplus column: another free
        descendant's own surplus is held, its height still moves). Ids increase from parent to
        child, so one pass over the descendants in id order suffices."""
        desc = {vid}; stack = [vid]
        while stack:
            for c in self.children.get(stack.pop(), []):
                if c not in desc: desc.add(c); stack.append(c)
        w = {vid: 1.0}
        for u in sorted(desc):
            if u == vid: continue
            row = self.v.loc[u]
            w[u] = 0.5 * (w.get(int(row.parent0), 0.0) + w.get(int(row.parent1), 0.0))
        return w

    def star_faces(self, vid: int) -> pd.DataFrame:
        """The cells the coefficient reaches: those with the vertex or a descendant carrying part of
        it as a corner, with the coefficient's weight at each corner (w00, w10, w01, w11)."""
        w = self.influence(vid); c = self.cells
        hit = np.zeros(len(c), bool)
        for k in self.CORNERS: hit |= c[f"v{k}"].isin(w).values
        out = c[hit].copy()
        for k in self.CORNERS: out[f"w{k}"] = out[f"v{k}"].map(w).fillna(0.0)
        return out

    def nodal_faces(self, vid: int) -> pd.DataFrame:
        c = self.cells
        hit = np.zeros(len(c), bool)
        for k in self.CORNERS: hit |= (c[f"v{k}"] == vid).values
        return c[hit]

    def bilinear(self, cell, *P):
        """(inside, the corner weights in corner order) of the points in the cell: bilinear in two
        dimensions, trilinear in three."""
        eps = 1e-12
        if self.dim == 2:
            px = np.asarray(P[0], float); py = np.asarray(P[1], float)
            inside = (px >= cell.x0 - eps) & (px <= cell.x1 + eps) & (py >= cell.y0 - eps) & (py <= cell.y1 + eps)
            u = np.clip((px - cell.x0) / (cell.x1 - cell.x0), 0, 1); t = np.clip((py - cell.y0) / (cell.y1 - cell.y0), 0, 1)
            return inside, ((1 - u) * (1 - t), u * (1 - t), (1 - u) * t, u * t)
        P = [np.asarray(q, float) for q in P]
        inside = np.ones(len(P[0]), bool); f = []
        for q, lo, hi in zip(P, self.lo_cols, self.hi_cols):
            a_, b_ = getattr(cell, lo), getattr(cell, hi)
            inside &= (q >= a_ - eps) & (q <= b_ + eps); f.append(np.clip((q - a_) / (b_ - a_), 0, 1))
        w = []
        for k in range(8):
            weight = f[0] if k & 1 else 1 - f[0]
            for d in (1, 2): weight = weight * (f[d] if (k >> d) & 1 else 1 - f[d])
            w.append(weight)
        return inside, tuple(w)

    def hat(self, cell, *P):
        inside, b = self.bilinear(cell, *P)
        if self.dim == 2:
            return inside, b[0] * cell.w00 + b[1] * cell.w10 + b[2] * cell.w01 + b[3] * cell.w11
        return inside, sum(b[k] * getattr(cell, f"w{name}") for k, name in enumerate(self.CORNERS))

    def star_pools(self, vid: int) -> pd.DataFrame:
        """Pools inside the star with the coefficient's hat value at each."""
        rows = []
        for _, cell in self.star_faces(vid).iterrows():
            inside, h = self.hat(cell, *self.coords(self.obs))
            sub = self.obs[inside].copy(); sub["hat"] = h[inside]
            rows.append(sub)
        pools = pd.concat(rows) if rows else self.obs.iloc[0:0].assign(hat=0.0)
        pools = pools.loc[~pools.index.duplicated()]
        pools = pools[pools.hat > 1e-9].copy()
        # the residual the gate scored: against the null surface, the surplus removed
        v = self.v.loc[vid]
        eta_null = pools.base.values + (pools.g_hat.values - float(v.surplus) * pools.hat.values * pools.uses_mesh.values) / self.height_scale
        p_null = 1 / (1 + np.exp(-eta_null)); I_null = pools.n.values * p_null * (1 - p_null)
        pools["r_null"] = (pools.k.values - pools.n.values * p_null) / np.sqrt(I_null)
        return pools

    def surface(self, *P, faces=None):
        """f in logits at points, multilinear on the given cells."""
        cells = self.cells if faces is None else faces
        out = np.full(len(P[0]), np.nan)
        for _, cell in cells.iterrows():
            inside, b = self.bilinear(cell, *P)
            inside &= np.isnan(out)
            if self.dim == 2:
                out[inside] = (b[0] * cell.h00 + b[1] * cell.h10 + b[2] * cell.h01 + b[3] * cell.h11)[inside]
            else:
                out[inside] = sum(b[k] * getattr(cell, f"h{name}") for k, name in enumerate(self.CORNERS))[inside]
        return out

    def column_at(self, vid: int, *P) -> np.ndarray:
        """The coefficient's hierarchical column at points (0 outside its cells)."""
        out = np.zeros(len(P[0])); done = np.zeros(len(P[0]), bool)
        for _, cell in self.star_faces(vid).iterrows():
            inside, h = self.hat(cell, *P)
            inside &= ~done
            out[inside] = h[inside]; done |= inside
        return out

    def months(self, dx): return dx * (self.wala_range[1] - self.wala_range[0])
    def points(self, dy): return dy * (self.wac_range[1] - self.wac_range[0])
    def balance(self, z): return float(np.exp(self.axis3_range[0] + z * (self.axis3_range[1] - self.axis3_range[0])))

    def geometry(self, vid: int, pools: pd.DataFrame) -> dict:
        """What is particular to a rectangular vertex: the segment it bisects and the cut through it,
        the move that admitted it, the gate's competing perpendicular cuts, the far ends of its
        chord, and its column's shape and nesting."""
        v = self.v.loc[vid]; g = dict(root=int(v.parent0) < 0)
        faces = self.star_faces(vid); own = self.nodal_faces(vid)
        g["extent"] = (self.months(faces.x1.max() - faces.x0.min()), self.points(faces.y1.max() - faces.y0.min()))
        g["aspect"] = (faces.x1.max() - faces.x0.min()) / (faces.y1.max() - faces.y0.min())
        if self.dim == 3: g["balance_range"] = (self.balance(faces.z0.min()), self.balance(faces.z1.max()))
        g["own_cells"] = len(own); g["reached_cells"] = len(faces) - len(own)
        desc = [u for u in self.influence(vid) if u != vid]
        g["descendants"] = len(desc)
        g["hanging_desc"] = sum(u in self.hanging for u in desc)
        g["hanging"] = vid in self.hanging
        free_desc = [u for u in desc if self.v.loc[u].free_coefficient == 1]
        g["free_desc"] = len(free_desc)
        # the free descendants' part of the column's variation: information-weighted, on the pools
        # that informed it, this coefficient's term against the sum of its free descendants' terms
        w = np.where(pools.in_fit.values, pools["weight"].values, 0.0)
        mine = float(v.surplus) * pools.hat.values
        theirs = np.zeros(len(pools))
        for u in free_desc:
            theirs += float(self.v.loc[u].surplus) * self.column_at(u, *self.coords(pools))
        a, b = float((w * mine ** 2).sum()), float((w * theirs ** 2).sum())
        g["desc_share"] = b / (a + b) if a + b > 0 else 0.0
        g["move"] = int(v.admit_move) if "admit_move" in v.index else -1
        g["splits"] = int(v.admit_splits) if "admit_splits" in v.index else 0
        if g["root"]:
            return g
        p0, p1 = self.v.loc[int(v.parent0)], self.v.loc[int(v.parent1)]
        axis = int(v.axis) if self.dim == 3 else (0 if abs(p1.x - p0.x) > abs(p1.y - p0.y) else 1)
        g["axis"] = axis
        if axis == 2:   # a cut across the balance axis: the segment in dollars of average balance
            g["segment"] = abs(self.balance(p1.z) - self.balance(p0.z))
            g["ends"] = sorted([self.balance(q.z) for q in (p0, p1)])
            g["chord"] = {}
            return g
        g["segment"] = self.months(abs(p1.x - p0.x)) if axis == 0 else self.points(abs(p1.y - p0.y))
        g["ends"] = sorted([self.wala_range[0] + self.months(q.x) if axis == 0 else self.wac_range[0] + self.points(q.y) for q in (p0, p1)])
        # the chord: the cut through the vertex, across its segment, as the admission left it (later
        # splits shorten it); its far ends are the midpoints of the split cells' opposite edges
        g["chord"] = {}
        if "admit_chord_lo" in v.index and v.admit_round >= 0:
            for side, u in ((False, int(v.admit_chord_lo)), (True, int(v.admit_chord_hi))):
                if u < 0: continue
                q = self.v.loc[u]
                g["chord"][side] = dict(at=(self.wac_range[0] + self.points(q.y)) if axis == 0 else (self.wala_range[0] + self.months(q.x)),
                                        free=bool(q.free_coefficient), hanging=u in self.hanging, id=u)
        # the gate's round: this vertex's rank, and the other way to split the same cells: their other
        # midline, whose ends are the midpoints of the cells' edges through the parents (the cells'
        # extent across the segment is the chord's, since the admission cuts only along it)
        if self.cand is not None and "admit_round" in v.index and v.admit_round >= 0:
            r = self.cand[self.cand["round"] == int(v.admit_round)]
            me = r[(np.abs(r.x - v.x) < 1e-12) & (np.abs(r.y - v.y) < 1e-12) & ((np.abs(r.x3 - v.z) < 1e-12) if self.dim == 3 else True)]
            if len(me):
                zc = abs(float(me.z.iloc[0]))
                g["round_size"] = len(r); g["rank"] = int((np.abs(r.z) > zc).sum()) + 1
                spots = []   # a promoted weld splits nothing: its cells are shorter than its segment
                for e in (g["chord"].values() if g["move"] != 1 else ()):
                    q = self.v.loc[e["id"]]
                    for par in (p0, p1):
                        spots.append((par.x, (v.y + q.y) / 2) if axis == 0 else ((v.x + q.x) / 2, par.y))
                hit = np.zeros(len(r), bool)
                for sx_, sy_ in spots:
                    hit |= (np.abs(r.x.values - sx_) < 1e-12) & (np.abs(r.y.values - sy_) < 1e-12)
                comp = r[hit & (r.axis.values != int(me.axis.iloc[0]))]
                g["competitors"] = len(comp)
                if len(comp):
                    b = comp.loc[comp.z.abs().idxmax()]
                    later = self.v[(np.abs(self.v.x - b.x) < 1e-12) & (np.abs(self.v.y - b.y) < 1e-12) & (self.v.free_coefficient == 1)]
                    g["rival"] = dict(z=abs(float(b.z)), at=(self.wala_range[0] + self.months(b.x), self.wac_range[0] + self.points(b.y)),
                                      admitted=bool(b.admitted), selected=bool(b.selected),
                                      later=int(later.admit_round.iloc[0]) if len(later) and later.admit_round.iloc[0] > v.admit_round else None,
                                      near_tie=abs(float(b.z)) >= 0.9 * zc)
        return g

    # ---- the pool effect --------------------------------------------------------------------
    _GH = np.polynomial.hermite.hermgauss(24)

    @classmethod
    def centred_shift(cls, eta, s2):
        """The shift c per pool such that a normal effect N(-c, s2) on the log-odds leaves the
        pool's expected rate at expit(eta). The surface is fitted to counts, so it estimates the
        marginal rate, the mean over pools; an uncentred effect would raise that mean by about
        s2/2 in logit and count the variance twice. Newton on c, Gauss-Hermite for the mean."""
        x, w = cls._GH; eta = np.asarray(eta, float)
        if s2 <= 0: return np.zeros(len(eta))
        u = np.sqrt(2 * s2) * x; target = 1 / (1 + np.exp(-eta)); c = np.full(len(eta), s2 / 2)
        for _ in range(12):
            q = 1 / (1 + np.exp(-(eta[:, None] - c[:, None] + u[None, :])))
            mean = np.sum(q * w[None, :], 1) / np.sqrt(np.pi); slope = -np.sum(q * (1 - q) * w[None, :], 1) / np.sqrt(np.pi)
            c = c - (mean - target) / slope
        return c

    VARIANCE_SAMPLE = 1200   # pools beyond this add precision the band cannot show

    @classmethod
    def pool_variance(cls, n, k, eta) -> float:
        """The variance of a normal, mean-preserving pool effect on the log-odds, by maximum
        marginal likelihood over the given pools at the fitted surface eta (Gauss-Hermite, 24
        nodes; golden section on log s2). The moment estimator from Pearson residuals fails on
        rare events, where the many zero-event pools pull it negative; this one does not.

        Above VARIANCE_SAMPLE pools it is fitted on a deterministic stride through them: the
        estimate's relative error is about sqrt(2/m), so 1200 pools fix the standard deviation
        to about 2%, less than a band's line width. analysis/starcore.js takes the same stride,
        so the static and the interactive pages agree."""
        n = np.asarray(n, float); k = np.asarray(k, float); eta = np.asarray(eta, float)
        if len(n) > cls.VARIANCE_SAMPLE:
            stride = -(-len(n) // cls.VARIANCE_SAMPLE)
            n, k, eta = n[::stride], k[::stride], eta[::stride]
        x, w = cls._GH
        def nll(log_s2):
            s2 = np.exp(log_s2); u = np.sqrt(2 * s2) * x; e = (eta - cls.centred_shift(eta, s2))[:, None] + u[None, :]
            ll = k[:, None] * e - n[:, None] * np.log1p(np.exp(e)); m = ll.max(1, keepdims=True)
            return -np.sum(m[:, 0] + np.log(np.sum(np.exp(ll - m) * w[None, :], 1) / np.sqrt(np.pi)))
        a, b = np.log(1e-4), np.log(9.0); g = (np.sqrt(5) - 1) / 2
        c_, d_ = b - g * (b - a), a + g * (b - a); fc, fd = nll(c_), nll(d_)
        for _ in range(60):
            if fc < fd: b, d_, fd = d_, c_, fc; c_ = b - g * (b - a); fc = nll(c_)
            else: a, c_, fc = c_, d_, fd; d_ = a + g * (b - a); fd = nll(d_)
        return float(np.exp((a + b) / 2))

    # ---- accounting ---------------------------------------------------------------------
    def card(self, vid: int, pools: pd.DataFrame) -> dict:
        v = self.v.loc[vid]
        parents = [int(v.parent0), int(v.parent1)]
        null_h = float(v.height - v.surplus)   # the height at zero surplus: the parents' average (a root's is 0)
        train = pools.is_train.values == 1
        in_fit = pools.in_fit.values
        w = np.where(in_fit, pools["weight"].values, 0.0); x = pools.hat.values   # only the pools that informed the surface carry weight
        xTx = float((w * x * x).sum())
        share = w * x * x / xTx if xTx > 0 else np.zeros(len(pools))
        n, k, p = pools.n.values, pools.k.values, pools.p_hat.values
        p_off = np.where(pools.uses_mesh.values, 1 / (1 + np.exp(-pools.base.values)), p)
        def X(pp):
            I = n * pp * (1 - pp); r2 = (k - n * pp) ** 2 / I
            return float(((r2 - 1)[train]).sum() / I[train].sum())
        T = X(p_off); Xm = X(p); P = float(np.average(pools.s2.values[train], weights=(n * p * (1 - p))[train]))
        # what the data alone would give the surplus: one Newton step from the fit on the training
        # pools under the hat, no prior; each pool weighted I/(1 + I s2) with the star's own pool
        # variance s2, so the se counts the pool effect (a binomial-only se overstates the evidence
        # of a few large, overdispersed pools)
        eta_f = pools.base.values + pools.g_hat.values / self.height_scale
        s2_star = self.pool_variance(n[in_fit], k[in_fit], eta_f[in_fit]) if in_fit.sum() > 2 else 0.0
        I_ = n * p * (1 - p); wt = np.where(in_fit, I_ / (1 + I_ * s2_star), 0.0)
        xtwx = float(np.sum(wt * x * x))
        num = float(np.sum(wt * x * (k - n * p) / np.maximum(I_, 1e-12)))
        data_move = float(v.surplus) / self.height_scale + (num / xtwx if xtwx > 0 else 0.0)
        data_se = 1 / np.sqrt(xtwx) if xtwx > 0 else float("inf")
        prior_sd = 1 / np.sqrt(float(v.lambda_v)) if float(v.lambda_v) > 0 else float("inf")
        # held-out pools under the vertex: what the split scores
        held = ~train
        held_dev = float(deviance(k[held], n[held], p[held]).mean()) if held.any() else float("nan")
        # pools stacked on one covariate point
        pt = pd.Series(list(zip(pools.wala_m.values, np.round(pools.wac_pct.values, 6))))
        stacks = pt.value_counts()
        return dict(id=vid, depth=int(v.depth), parents=parents, height_logit=v.height / self.height_scale,
                    rate=1 / (1 + np.exp(-(self.pooled_logit + v.height / self.height_scale))),
                    surplus_logit=float(v.surplus) / self.height_scale, null_h=null_h,
                    pools=len(pools), pools_train=int(in_fit.sum()), xTx=xTx, keff=float(v.current_keff), keff_share=1.0 / float((share * share).sum()) if xTx > 0 else 0.0,
                    max_share=float(share.max()), max_share_pid=int(pools.pid.values[share.argmax()]),
                    max_share_n=int(n[share.argmax()]), admitted=bool(v.gate_admitted), free=bool(v.free_coefficient),
                    lambda_v=float(v.lambda_v), T=T, S=T - Xm, P=P, R=Xm - P, share=share,
                    data_move=data_move, data_se=data_se, prior_sd=prior_sd,
                    held_pools=int(held.sum()), held_obs=100 * k[held].sum() / max(n[held].sum(), 1),
                    held_fit=100 * float(np.average(p[held], weights=n[held])) if held.any() else float("nan"), held_dev=held_dev,
                    stack_n=int(stacks.iloc[0]) if len(stacks) else 0, stack_at=stacks.index[0] if len(stacks) else None)

    def rank(self) -> pd.DataFrame:
        """The gate's admissions with their cards: the decisions a star plot re-litigates (the roots'
        and the backbone's columns span the square)."""
        rows = []
        for vid in self.v[(self.v.free_coefficient == 1) & (self.v.gate_admitted == 1)].id:
            pools = self.star_pools(vid)
            if len(pools) < 5: continue
            c = self.card(vid, pools); c.pop("share")
            rows.append(c)
        return pd.DataFrame(rows)

    # ---- drawing ------------------------------------------------------------------------
    STRIP = 0.35   # a section takes the pools within this many half-edge lengths of its line

    def profile(self, ax, vid, pools, direction, label, bins=8, color="#2c3e50", s2_type=None, half=None, strip_unit=None, reach=None):
        """One section through the star, everything relative to the null surface (the surplus
        removed): the null is the zero line, the red line is what the surplus adds at each
        bin's pools, and the points are the observed log-odds of the bin's pools above or below
        the null, one point per issue type. Equal-width bins on [-1, 1] in half-edge lengths,
        pools within the strip around the line only."""
        v = self.v.loc[vid]; d = direction / np.linalg.norm(direction)
        if half is None:
            half = np.linalg.norm(direction) / 2 if np.linalg.norm(direction) > 0 else 1.0
        # the strip is a fraction of the column's extent across the line (cells can be far from square);
        # in three dimensions a tube: the distance from the line
        if self.dim == 2:
            off = ((pools.sx.values - v.x) * (-d[1]) + (pools.sy.values - v.y) * d[0]) / (half if strip_unit is None else strip_unit)
        else:
            # a box around the line: on every other axis, within the column's reach from the vertex
            rel = np.column_stack([q - vq for q, vq in zip(self.coords(pools), self.at(v))])
            off = np.zeros(len(pools))
            for a_ in range(3):
                if abs(d[a_]) < 1e-12: off = np.maximum(off, np.abs(rel[:, a_]) / max(reach[a_], 1e-12) * self.STRIP)
        pools = pools[np.abs(off) <= self.STRIP]
        if len(pools) == 0:
            ax.text(0.5, 0.5, "no pools within the strip along this line", transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="0.4")
        if self.dim == 2:
            t = ((pools.sx.values - v.x) * d[0] + (pools.sy.values - v.y) * d[1]) / half
        else:
            t = sum((q - vq) * dq for q, vq, dq in zip(self.coords(pools), self.at(v), d)) / half
        surplus_units = float(v.surplus)
        f_fit = pools.g_hat.values; f_null = f_fit - surplus_units * pools.hat.values * pools.uses_mesh.values
        n, k = pools.n.values, pools.k.values; multi = pools.multi.values
        # the pool-to-pool variance behind the band: the design's law where there is one, else the
        # star's own, by marginal likelihood per issue type
        if self.tempered: s2 = pools.s2.values; band_label = "68% band for the bin: binomial + the pool law"
        elif s2_type is not None:
            s2 = np.where(multi, s2_type[True], s2_type[False])
            sds = f"sd {np.sqrt(s2_type[False]):.2f}" + (f" single / {np.sqrt(s2_type[True]):.2f} multi" if s2_type[True] != s2_type[False] else "") + " logit"
            band_label = f"68% band: binomial + pool effect, {sds}"
        else: s2 = np.zeros(len(pools)); band_label = "68% band for the bin: binomial"
        base = pools.base.values; eta_null_pool = base + f_null / self.height_scale
        edges = np.linspace(-1, 1, bins + 1); width = edges[1] - edges[0]
        # the exact tent on the line, faint: the surplus times the hat along the section
        tt = np.linspace(-1, 1, 201)
        line = [vq + tt * half * dq for vq, dq in zip(self.at(v), d)]
        in_domain = np.ones(len(tt), bool)
        for q in line: in_domain &= (q >= -1e-12) & (q <= 1 + 1e-12)
        hat_line = np.where(in_domain, 0.0, np.nan)   # none outside the square (the cube)
        for _, cell in self.star_faces(vid).iterrows():
            inside, h = self.hat(cell, *line); hat_line[inside] = h[inside]
        ax.plot(tt, surplus_units * hat_line / self.height_scale, "-", color="#c0392b", lw=1.6, alpha=0.7, zorder=2.6, label="surplus on the section line itself")
        ax.axhline(0, color=color, lw=1.2, ls=(0, (4, 2)), zorder=2.5, label="null surface (surplus removed)")
        info_total = None                      # the section's total weight, so areas compare within a panel
        for a_, b_ in zip(edges[:-1], edges[1:]):
            m = (t >= a_) & (t < b_) if b_ < 1 else (t >= a_) & (t <= b_)
            if m.sum() == 0: continue
            for is_multi, marker, dx in ((False, "o", -0.12 * width), (True, "^", 0.12 * width)):
                if info_total is None:
                    info_total = float(np.sum(pools["weight"].values))
                g = m & (multi == is_multi)
                if g.sum() == 0: continue   # zero-event bins stay: they are the ramp
                pk, pn = k[g].sum(), n[g].sum()
                obs = np.log((pk + 0.5) / (pn - pk + 0.5))
                # the null on the same scale as the observation: the log-odds of the summed expected counts
                p_null = 1 / (1 + np.exp(-eta_null_pool[g])); null_g = np.log(np.sum(n[g] * p_null) / (pn - np.sum(n[g] * p_null)))
                x_ = (a_ + b_) / 2 + dx
                expected = np.sum(n[g] / (1 + np.exp(-(base[g] + f_fit[g] / self.height_scale))))
                ax.plot([x_], [np.log(expected / (pn - expected)) - null_g], marker="_", color="#c0392b", ms=14, mew=2.6, zorder=3.8,
                        label="fit, same bin and issuer type" if "fit, same bin and issuer type" not in ax.get_legend_handles_labels()[1] else None)
                if "p_other" in pools:
                    e_o = np.sum(n[g] * pools.p_other.values[g])
                    ax.plot([x_], [np.log(e_o / (pn - e_o)) - null_g], marker="_", color="0.2", ms=12, mew=1.8, alpha=0.8, zorder=3.7,
                            label="compared run's fit" if "compared run's fit" not in ax.get_legend_handles_labels()[1] else None)
                if pk == 0:   # no events: the half-event correction would put the point logits above the null; show the count instead
                    nz = getattr(ax, "_zero_bins", 0); ax._zero_bins = nz + 1   # stagger the labels so adjacent empty bins stay legible
                    ax.annotate(f"{int(pn):,} loans, 0 events", (x_, 0.0), xytext=(0, 5 + 9 * (nz % 3)), textcoords="offset points", fontsize=6.5, color="0.35", ha="center", zorder=6)
                    continue
                # the model's 68% predictive band for this bin's observed log-odds: binomial counts at the
                # fitted surface with a normal pool effect per pool and the vertex's posterior; conditioned
                # on at least one event, as the display is; the count jittered by a half unit so the
                # percentiles of a discrete count are exact (a randomised quantile). Checked to cover
                # 68% of data drawn from the model itself; it is drawn where the model puts it, not
                # around the observed point, so the point either sits in it or does not.
                rng = np.random.default_rng(int(1e6 * (a_ + 2)) + int(is_multi)); S = 4000
                eta_fit = base[g] + f_fit[g] / self.height_scale; nn = n[g].astype(int)
                sd_v = np.sqrt(v.sigma_pooled) / self.height_scale if v.sigma_pooled < 1e100 else 0.0
                shift = np.zeros(g.sum())   # the effect centred so each pool's expected rate stays at the fitted surface
                for s2v in np.unique(s2[g]):
                    sel = s2[g] == s2v; shift[sel] = self.centred_shift(eta_fit[sel], float(s2v))
                eta = (eta_fit - shift)[None, :] + rng.normal(0, 1, (S, 1)) * sd_v * (pools.hat.values[g] * pools.uses_mesh.values[g])[None, :] + rng.normal(0, 1, (S, g.sum())) * np.sqrt(s2[g])[None, :]
                kk = rng.binomial(nn[None, :].repeat(S, 0), 1 / (1 + np.exp(-eta))).sum(1).astype(float)
                kk = kk[kk >= 1]
                if len(kk) >= 50:
                    kk = kk + rng.uniform(-0.5, 0.5, len(kk)); pp = np.clip((kk + 0.5) / (pn + 1), 1e-12, 1 - 1e-12)
                    lo, hi = np.percentile(np.log(pp / (1 - pp)), [16, 84])
                    ax.plot([x_, x_], [lo - null_g, hi - null_g], color="#e67e22", lw=3.0, alpha=0.55, solid_capstyle="butt", zorder=3.5,
                            label=band_label if not ax.get_legend_handles_labels()[1].count(band_label) else None)
                y_ = obs - null_g
                # Count residual against the fitted prediction. Effective information is
                # a fitting weight, not the variance of an aggregate event count.
                p_fit = 1 / (1 + np.exp(-eta_fit))
                variance_g = float(np.sum(n[g] * p_fit * (1 - p_fit)))
                info_g = float(np.sum(pools["weight"].values[g]))
                z_g = (pk - float(np.sum(n[g] * p_fit))) / np.sqrt(variance_g) if variance_g > 0 else 0.0
                ax.plot([x_], [y_], marker, color=plt.get_cmap("RdBu_r")((np.clip(z_g, -3, 3) + 3) / 6),
                        markeredgecolor="0.25", markeredgewidth=0.6, zorder=4,
                        ms=float(np.sqrt(4.0 + 150.0 * info_g / max(info_total, 1e-12))),
                        label=("multi-issuer" if is_multi else "single-issuer") + " pools, observed" if not ax.get_legend_handles_labels()[1].count(("multi-issuer" if is_multi else "single-issuer") + " pools, observed") else None)

        ax.axvline(0, color="0.6", lw=0.6); ax.set_ylabel("log-odds relative to the null"); ax.grid(alpha=0.25)
        ax.set_xlim(-1.08, 1.08); ax.set_xlabel(label)

    def card_lines(self, vid, full, pools, c, v):
        """The card's lines, (text, bold): the vertex, its admission, its cut, its column, what it
        contributes and the evidence under it."""
        tr_ = pools.is_train.values == 1
        def grp(m):
            n_, k_ = pools.n.values[m], pools.k.values[m]
            return (int(m.sum()), int(n_.sum()), 100 * k_.sum() / max(n_.sum(), 1), 100 * np.average(pools.p_hat.values[m], weights=n_) if m.sum() else float("nan"))
        pa = [self.address(self.v.loc[q]) for q in c["parents"] if q >= 0]
        wala_v = self.wala_range[0] + v.x * (self.wala_range[1] - self.wala_range[0]); wac_v = self.wac_range[0] + v.y * (self.wac_range[1] - self.wac_range[0])
        where = f", average balance ${self.balance(v.z):,.0f}" if self.dim == 3 else ""
        lines = [(f"{self.address(v)}   parents A {pa[0]}, A' {pa[1]}" if len(pa) == 2 else f"{self.address(v)}   a root corner", True),
                 (f"  at loan age {wala_v:.1f} months, coupon {wac_v:.3f}%{where}; depth {c['depth']}", False), ("", False)]
        if "admit_round" in v.index and v.admit_round >= 0:
            root = v.admit_signed_root if "admit_signed_root" in v.index and np.isfinite(v.admit_signed_root) else v.admit_beta
            direction = "lower" if root < 0 else "higher"
            gain = f"     exact gain {v.admit_gain:.1f} nats" if "admit_gain" in v.index and np.isfinite(v.admit_gain) else ""
            margin = abs(v.admit_z) - v.admit_c
            edge = "   ON THE THRESHOLD" if margin < 0.05 * max(v.admit_c, 1e-9) else ""
            lines += [(f"ADMISSION   round {int(v.admit_round)}", True),
                      (f"  PASS   |score| {abs(v.admit_z):.2f} > {v.admit_c:.2f} (margin {margin:.2f}){edge}     direction: {direction} than the round-{int(v.admit_round)} surface{gain}", False)]
        elif c["free"] and v.depth <= 1:
            lines += [("ADMISSION   free by construction: " + (f"a corner of the {self.domain}" if v.depth == 0 else "the depth-1 backbone"), True)]
        elif c["free"]:
            lines += [("ADMISSION   before this run recorded scores", True)]
        else:
            lines += [("ADMISSION   not admitted; no coefficient here", True)]
        g = self.geometry(vid, full)
        if g.get("rank"):
            lines += [(f"  rank {g['rank']} of {g['round_size']} candidates in its round by |score|", False)]
        if g["move"] >= 0:
            move = {0: f"a new midpoint: {g['splits']} cell{'s' if g['splits'] != 1 else ''} split to make it a corner",
                    1: "an existing vertex, already a corner of every cell around it: promoted, nothing split",
                    2: f"an existing vertex hanging on a coarser edge: released, {g['splits']} cell{'s' if g['splits'] != 1 else ''} split"}[g["move"]]
            lines += [(f"  move: {move}", False)]
        lines += [("", False), ("THE CUT", True)]
        if g["root"]:
            lines += [(f"  none: a corner of the {self.domain}, where the surface starts", False)]
        elif g["axis"] == 2:
            lines += [(f"  bisects a ${g['segment']:,.0f} segment of average balance (${g['ends'][0]:,.0f} to ${g['ends'][1]:,.0f}); the cut runs across loan age and coupon", False)]
        else:
            what, unit, other = ("loan age", "months", "coupon") if g["axis"] == 0 else ("coupon", "points", "loan age")
            if self.dim == 3: other += " and balance"
            fmt = (lambda t: f"{t:.1f}") if g["axis"] == 0 else (lambda t: f"{t:.3f}")
            lines += [(f"  bisects a {g['segment']:.{1 if g['axis'] == 0 else 3}f}-{unit[:-1]} segment of {what} ({fmt(g['ends'][0])} to {fmt(g['ends'][1])}); the cut runs across {other}", False)]
            if "rival" in g:
                b = g["rival"]
                fate = ("also admitted" if b["admitted"] else "selected, rejected on re-profile" if b["selected"] else
                        f"not selected; admitted in round {b['later']}" if b["later"] is not None else "not selected")
                lines += [(f"  the other way to split the same cells: {g['competitors']} candidate{'s' if g['competitors'] != 1 else ''}, best |score| {b['z']:.2f} at age {b['at'][0]:.1f}, coupon {b['at'][1]:.3f}%, {fate}"
                           + ("   NEAR TIE" if b["near_tie"] else ""), False)]
            elif g["move"] == 1:
                lines += [("  a promotion splits no cell, so there is no other way to split", False)]
            elif g.get("rank") and g["chord"]:
                lines += [("  the other way to split the same cells: no candidate in that round", False)]
            ends = []
            for side, name in ((False, "below" if g["axis"] == 0 else "left"), (True, "above" if g["axis"] == 0 else "right")):
                e = g["chord"].get(side)
                if e is None: ends.append(f"{name}: none (the square's edge)"); continue
                at = f"{e['at']:.3f}%" if g["axis"] == 0 else f"{e['at']:.1f} months"
                ends.append(f"{name} at {at}, " + ("free" if e["free"] else "constrained") + (", hanging on a coarser edge" if e["hanging"] else ""))
            if g["chord"]:
                lines += [("  the cut's far ends at admission, now: " + "; ".join(ends), False)]
        shape = (f"  {g['extent'][0]:.1f} months x {g['extent'][1]:.3f} points x ${g['balance_range'][0]:,.0f} to ${g['balance_range'][1]:,.0f} of balance;"
                 if self.dim == 3 else f"  {g['extent'][0]:.1f} months x {g['extent'][1]:.3f} points, {g['aspect']:.2g}:1 in the square's scaled units;")
        lines += [("", False), ("COLUMN   where the coefficient moves the surface", True),
                  (shape +
                   f" {g['own_cells']} own cell{'s' if g['own_cells'] != 1 else ''}"
                   + (f" + {g['reached_cells']} reached through {g['descendants']} descendant{'s' if g['descendants'] != 1 else ''}" if g["descendants"] else ", no descendants"), False)]
        if g["descendants"]:
            free_part = (f"{g['free_desc']} free descendant{'s' if g['free_desc'] != 1 else ''} carr{'ies' if g['free_desc'] == 1 else 'y'} {100 * g['desc_share']:.0f}% of the column's variation"
                         if g["free_desc"] else "no free descendants: the column is this coefficient's alone")
            lines += [(f"  {free_part}; {g['hanging_desc']} descendant{'s' if g['hanging_desc'] != 1 else ''} hang{'s' if g['hanging_desc'] == 1 else ''} on a coarser edge"
                       + ("; this vertex hangs too" if g["hanging"] else ""), False)]
        lines += [("", False), ("CONTRIBUTION   relative to removing the coefficient", True)]
        if c["free"]:
            lines += [(f"  correction {c['surplus_logit']:+.3f} logit     odds x{np.exp(c['surplus_logit']):.2f}", False)]
            floor = c["prior_sd"] <= 0.0051
            if g["root"]:
                lines += [(f"  prior: sd {np.sqrt(K_ROOT_PRIOR_VARIANCE):.3f} logit around the baseline (a corner of the {self.domain})", False)]
            else:
                seg = f"{g['segment']:.1f}-month" if g["axis"] == 0 else f"${g['segment']:,.0f}-balance" if g["axis"] == 2 else f"{g['segment']:.3f}-point"
                lines += [(f"  prior sd at depth {c['depth']} ({seg} segment): {c['prior_sd']:.3f} logit" + ("   AT THE FLOOR: the prior pins this coefficient" if floor else ""), False)]
            lines += [
                      (f"  data alone (no prior; pool effect in the se): {c['data_move']:+.2f} logit (se {c['data_se']:.2f}); the fit keeps {c['surplus_logit']:+.3f}", False)]
        else:
            lines += [("  none: the height here is the parents' average", False)]
        types = []
        for name, m in (("single", tr_ & ~pools.multi.values), ("multi", tr_ & pools.multi.values)):
            if m.sum():
                np_, nl, ob, ft = grp(m); types.append(f"{name} {np_} pools {ob:.2f}% obs / {ft:.2f}% fit")
        if types: lines += [("  under it: " + "; ".join(types), False)]
        if self.combined and self.combined.get("msf_baseline") == "direct":
            lines += [("  Direct M-SF: mesh contribution is zero (comparison only).", False)]
        lines += [("", False), ("EVIDENCE   training pools under the coefficient", True),
                  (f"  {c['pools_train']} pools     effective {c['keff_share']:.1f}     largest {100 * c['max_share']:.0f}% (pool {c['max_share_pid']}, {c['max_share_n']:,} loans)", False)]
        if c["max_share"] >= 1 / 3:
            lines += [("  one pool holds more than a third of the evidence", False)]
        elif c["keff_share"] < 3:
            lines += [("  a few pools hold most of the evidence", False)]
        if c["stack_n"] >= 3:
            sw, sc_ = c["stack_at"]
            lines += [(f"  {c['stack_n']} pools share one covariate point (age {sw:g}, coupon {sc_:.3f}%)", False)]
        if c["held_pools"]:
            lines += [(f"  held-out under it: {c['held_pools']} pools, {c['held_obs']:.2f}% obs / {c['held_fit']:.2f}% fit, mean deviance {c['held_dev']:.2f}", False)]
        if self.compare is not None:
            po = full.p_other.values; nn_ = full.n.values; tr_f = full.is_train.values == 1
            f_this = 100 * np.average(full.p_hat.values, weights=nn_); f_other = 100 * np.average(po, weights=nn_)
            lines += [(f"  compared run ({self.compare_label}): {f_other:.2f}% fit under the star against {f_this:.2f}% here", False)]
            if (~tr_f).any():
                lines += [(f"  held-out mean deviance: compared {deviance(full.k.values[~tr_f], nn_[~tr_f], po[~tr_f]).mean():.2f}, here {c['held_dev']:.2f}", False)]
        return lines

    def at(self, u):
        """A vertex's unit coordinates, one per axis."""
        return (u.x, u.y, u.z) if self.dim == 3 else (u.x, u.y)

    @property
    def domain(self):
        return "cube" if self.dim == 3 else "square"

    def address(self, u):
        """A vertex's dyadic address; in three dimensions the balance coordinate too."""
        if self.dim == 2:
            return address_name(u.x, u.y)
        for L in range(31):
            s_ = 2.0 ** L; tol = min(max(s_ * 2e-10, 1e-9), 1e-3)
            a, b, z = round(u.x * s_), round(u.y * s_), round(u.z * s_)
            if abs(u.x * s_ - a) < tol and abs(u.y * s_ - b) < tol and abs(u.z * s_ - z) < tol:
                return f"L{L} ({a}, {b}, {z})"
        return f"({u.x:.6g}, {u.y:.6g}, {u.z:.6g})"

    def page(self, vid: int, out: Path, title: str = "", show_held_out: bool = False):
        full = self.star_pools(vid); c = self.card(vid, full); v = self.v.loc[vid]
        keep = np.ones(len(full), bool) if show_held_out else (full.is_train.values == 1)   # held-out pools return with --held-out
        pools = full[keep]; share_plot = c["share"][keep]
        faces = self.star_faces(vid)
        fig = plt.figure(figsize=(16, 13.5), dpi=130)
        gs = fig.add_gridspec(3, 3, width_ratios=[2.2, 1, 1], height_ratios=[2.3, 1, 1], wspace=0.28, hspace=0.5)
        ax = fig.add_subplot(gs[:, 0])
        # the plane: star faces shaded, all faces in a window, pools of the star
        shaded = set()
        for _, f in faces.iterrows():
            own = vid in (f.v00, f.v10, f.v01, f.v11)
            label = ("the star: cells with this vertex as a corner" if own else "cells it reaches through its descendants")
            ax.fill([f.x0, f.x1, f.x1, f.x0], [f.y0, f.y0, f.y1, f.y1], color="#e4e4e4" if own else "#f0f0f0", zorder=0,
                    label=None if label in shaded else label)
            shaded.add(label)
        bx0, bx1 = faces.x0.min(), faces.x1.max()
        by0, by1 = faces.y0.min(), faces.y1.max()
        padx, pady = 0.6 * (bx1 - bx0) + 0.01, 0.6 * (by1 - by0) + 0.01
        x0w, x1w, y0w, y1w = max(bx0 - padx, 0), min(bx1 + padx, 1), max(by0 - pady, 0), min(by1 + pady, 1)
        win = self.cells[(self.cells.x1 > x0w) & (self.cells.x0 < x1w) & (self.cells.y1 > y0w) & (self.cells.y0 < y1w)]
        # the proxy's artifact region, when the design has one
        x_art = (self.artifact_months - self.wala_range[0]) / (self.wala_range[1] - self.wala_range[0]) if self.artifact_months > 0 else -1.0
        if x_art > x0w:
            ax.axvspan(x0w, min(x_art, x1w), facecolor="0.92", edgecolor="0.75", hatch="//", linewidth=0, zorder=0.2, alpha=0.6)
        for _, f in win.iterrows():
            ax.plot([f.x0, f.x1, f.x1, f.x0, f.x0], [f.y0, f.y0, f.y1, f.y1, f.y0], color="#7f9db9", lw=0.5, zorder=1)
        # neighbours: every pool in the window outside the star, small and faint, same colour scale
        o = self.obs if show_held_out else self.obs[self.obs.is_train == 1]
        inwin = (o.sx >= x0w) & (o.sx <= x1w) & (o.sy >= y0w) & (o.sy <= y1w) & ~o.pid.isin(pools.pid)
        ax.scatter(o.sx[inwin], o.sy[inwin], s=6, c=np.clip(o.resid[inwin], -3, 3), cmap="RdBu_r", vmin=-3, vmax=3,
                   marker="o", edgecolors="none", alpha=0.35, zorder=2)
        share = share_plot   # marker area: share of the fit's information; pools scored but not fitted (a combined run's giants) at a fixed size
        size = np.where(pools.in_fit.values, 10 + 1600 * share, np.where(pools.is_train.values == 1, 48, np.where(pools.multi.values, 48, 28)))
        r = np.clip(pools.resid.values, -3, 3)   # the plane shows the residual after the fit; the sections keep the null view
        train = pools.is_train.values == 1
        order = np.argsort(-size)   # large first, so small markers stay visible
        multi = pools.multi.values; tlab = "multi-issuer" + (" (proxy: >= 1,000 loans)" if self.type_is_proxy else "")
        groups = [(train & ~multi, "o", "-", "single-issuer pool"), (train & multi, "^", "-", tlab + " pool")]
        if show_held_out: groups += [(~train & ~multi, "o", "--", "held-out pool (open, residual on the ring)"), (~train & multi, "^", "--", None)]
        for m, marker, ls, lab in groups:
            sel = order[m[order]]
            if len(sel) == 0: continue
            if ls == "--":   # held out: no face, the residual colours the ring
                ring = plt.get_cmap("RdBu_r")((r[sel] + 3) / 6)
                ax.scatter(pools.sx.values[sel], pools.sy.values[sel], s=size[sel], facecolors="none", edgecolors=ring, linewidths=1.3,
                           marker=marker, zorder=3.1, label=lab)
                continue
            sc = ax.scatter(pools.sx.values[sel], pools.sy.values[sel], s=size[sel], c=r[sel], cmap="RdBu_r", vmin=-3, vmax=3,
                            marker=marker, edgecolors="0.25", linewidths=0.4, zorder=3, label=lab)
        # stacks: several pools at one covariate point draw as one marker; say how many
        pts = pd.DataFrame(dict(x=np.round(pools.sx.values, 9), y=np.round(pools.sy.values, 9), tr=train)).groupby(["x", "y"]).tr.agg(["size", "sum"])
        # the ten largest stacks only: a wide column's plane would drown in labels
        for (sx_, sy_), row in pts[pts["size"] >= 3].sort_values("size", ascending=False).head(10).iterrows():
            ax.annotate(f"{int(row['size'])} pools" + (f" ({int(row['sum'])} train)" if show_held_out else ""), (sx_, sy_), xytext=(7, -10),
                        textcoords="offset points", fontsize=7, color="0.15", zorder=9,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.6", lw=0.5, alpha=0.85))
        if self.tempered:
            I_train = np.where(train, pools["info"].values, 0.0); ishare = I_train / I_train.sum()
            ring = 10 + 1600 * ishare
            big = ishare > 0.02
            ax.scatter(pools.sx.values[big], pools.sy.values[big], s=ring[big], facecolors="none", edgecolors="k", linewidths=1.0,
                       linestyle="--", zorder=3.2, label="ring: the pool's share of the untempered information")
        for s_ in (0.01, 0.10, 0.25):
            ax.scatter([], [], s=10 + 1600 * s_, c="w", edgecolors="0.25", linewidths=0.4, label=f"{100 * s_:.0f}% of the information")
        ax.plot([], [], color="#7f9db9", lw=1.0, label="cell edge")
        ax.plot([], [], color="0.5", lw=2.0, label="parent segment edge, heavier when coarser (the child is its midpoint)")
        ax.plot([], [], color="#1f77b4", lw=1.6, label="its parents -> this vertex")
        cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.035, pad=0.07, aspect=40)
        cb.set_label("residual after the refit (red: more events than fitted); area: share of the vertex's information", fontsize=9)
        # vertices in the window
        vs = self.v
        vs = vs[(vs.x >= win.x0.min()) & (vs.x <= win.x1.max()) & (vs.y >= win.y0.min()) & (vs.y <= win.y1.max())]
        free = vs.free_coefficient == 1
        ax.scatter(vs.x[~free], vs.y[~free], marker="D", s=30, c="#2e8b57", edgecolors="k", linewidths=0.5, zorder=4, label="constrained vertex (parents' average)")
        ax.scatter(vs.x[free], vs.y[free], marker="D", s=40, c="#f1c40f", edgecolors="k", linewidths=0.6, zorder=5, label="free coefficient")
        ax.scatter([v.x], [v.y], marker="D", s=55, c="#f1c40f" if c["free"] else "#2e8b57", edgecolors="k", linewidths=1.8, zorder=8, label="this vertex (heavy edge)")
        # parent -> child arrows: thin grey for every vertex in the window, the highlighted
        # vertex's own lineage in black (grandparents -> parents -> vertex), parents ringed red
        def arrow(child, parent, color, lw, z):
            pv, cv = self.v.loc[parent], self.v.loc[child]
            if not (x0w <= pv.x <= x1w and y0w <= pv.y <= y1w): return   # keep the plane clean: parents outside the window are not drawn
            ax.annotate("", xy=(cv.x, cv.y), xytext=(pv.x, pv.y), zorder=z, annotation_clip=True,
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=4, shrinkB=11, mutation_scale=10 if lw < 1 else 16, clip_on=True))
        # each vertex's parent edge, weighted by depth: the coarser the edge, the heavier the line, so a
        # child is always the midpoint of a heavier line than its own and the generation order is readable
        dmax = int(vs.depth.max()) if len(vs) else 1
        for _, u in vs.sort_values("depth").iterrows():
            p0, p1 = int(u.parent0), int(u.parent1)
            if p0 >= 0 and p1 >= 0 and p0 in self.v.index and p1 in self.v.index:
                lw = 0.6 + 2.4 * (dmax - int(u.depth)) / max(dmax - 1, 1)
                ax.plot([self.v.loc[p0].x, self.v.loc[p1].x], [self.v.loc[p0].y, self.v.loc[p1].y], color="0.5", lw=lw, alpha=0.55, zorder=1.5, clip_on=True)
        for p in c["parents"]:
            if p < 0: continue
            arrow(vid, p, "#1f77b4", 1.6, 3.6)   # the parents' arrows run along section A-A', in its colour
        ax.set_xlim(x0w, x1w); ax.set_ylim(y0w, y1w)
        # ticks at the star's own vertices, so the labels are the mesh's coordinates
        own = self.nodal_faces(vid)
        xt = np.unique(np.round(np.r_[own.x0, own.x1], 9)); yt = np.unique(np.round(np.r_[own.y0, own.y1], 9))
        ax.set_xticks(xt); ax.set_xticklabels([f"{self.wala_range[0] + t * (self.wala_range[1] - self.wala_range[0]):.1f}" for t in xt], fontsize=8); ax.set_xlim(x0w, x1w)
        ax.set_yticks(yt); ax.set_yticklabels([f"{self.wac_range[0] + t * (self.wac_range[1] - self.wac_range[0]):.2f}" for t in yt], fontsize=8); ax.set_ylim(y0w, y1w)
        ax.set_xlabel("loan age, months" + (f" (hatched: under {self.artifact_months:g} months, the proxy's artifact region)" if self.artifact_months > 0 else "")); ax.set_ylabel("coupon, percent")
        ax.set_title((f"{title}   " if title else "") + f"vertex {address_name(v.x, v.y)} (id {vid}), {'free coefficient' if c['free'] else 'constrained vertex'}", fontsize=10)
        # the two section lines, drawn where they cross the star, labelled to match the profiles
        p0, p1 = c["parents"]
        along = np.array([self.v.loc[p1].x - self.v.loc[p0].x, self.v.loc[p1].y - self.v.loc[p0].y]) if p0 >= 0 else np.array([1.0, 0.0])
        perp = np.array([-along[1], along[0]]) if p0 >= 0 else np.array([0.0, 1.0])
        # orient the sections so A and B sit left / below and A' and B' right / above
        if along[0] < -1e-12 or (abs(along[0]) <= 1e-12 and along[1] < 0): along = -along
        if perp[1] < -1e-12 or (abs(perp[1]) <= 1e-12 and perp[0] < 0): perp = -perp
        # the section lengths: along the segment its half-length (the parents at -1 and +1), across it
        # the vertex's own cells' extent in that direction (the hat's edge at -1 and +1); a root
        # corner has no segment, so both come from its own cells
        # the column's reach from the vertex in each direction, over every cell it touches
        extent_x = float(np.maximum(np.abs(faces.x0 - v.x), np.abs(faces.x1 - v.x)).max()) if len(faces) else 0.5
        extent_y = float(np.maximum(np.abs(faces.y0 - v.y), np.abs(faces.y1 - v.y)).max()) if len(faces) else 0.5
        along_is_x = abs(along[0]) >= abs(along[1])
        half = np.linalg.norm(along) / 2 if p0 >= 0 else (extent_x if along_is_x else extent_y)
        half_across = extent_y if along_is_x else extent_x
        # what sits at each section end: a free coefficient (yellow), a constrained vertex (green), or no vertex (white)
        def vertex_type_at(ex, ey):
            hit = self.v[np.hypot(self.v.x - ex, self.v.y - ey) < 1e-6]
            if len(hit) == 0: return "no vertex", "white"
            return ("free", "#f1c40f") if hit.free_coefficient.iloc[0] == 1 else ("constrained", "#2e8b57")
        end_types = {}
        for vec, name, color, h_ in ((along, "A", "#1f77b4", half), (perp, "B", "#8e44ad", half_across)):
            u = vec / np.linalg.norm(vec)
            ends = [(v.x - u[0] * h_, v.y - u[1] * h_), (v.x + u[0] * h_, v.y + u[1] * h_)]
            ends = [(min(max(ex, 0.0), 1.0), min(max(ey, 0.0), 1.0)) for ex, ey in ends]   # the square's edge
            ax.plot([ends[0][0], ends[1][0]], [ends[0][1], ends[1][1]], color=color, lw=1.4, zorder=3.5)   # above the pools, beneath the vertices
            for (ex, ey), lab in zip(ends, (name, name + "'")):
                kind, fc = vertex_type_at(ex, ey)
                if abs(ex - v.x) < 1e-12 and abs(ey - v.y) < 1e-12: kind = "the square's edge"   # the line leaves the square at the vertex
                end_types[lab] = kind
                ax.annotate(lab, (ex, ey), xytext=(6, 6), textcoords="offset points", color=color, fontsize=10, fontweight="bold",
                            ha="left", va="bottom", zorder=7, annotation_clip=True,
                            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
        # the legend in four columns, one per kind of mark: pools, marker area, mesh lines, vertices
        handles, labels = ax.get_legend_handles_labels(); by_label = dict(zip(labels, handles))
        tests = [lambda l: "pool" in l, lambda l: "% of the information" in l, lambda l: "edge" in l or "cells" in l, lambda l: True]
        groups = [[] for _ in tests]
        for l in labels:   # each label to the first column whose test it passes
            groups[next(i for i, t in enumerate(tests) if t(l))].append(l)
        rows = max(len(g) for g in groups)
        from matplotlib.patches import Patch
        spacer = Patch(facecolor="none", edgecolor="none")
        hs, ls = [], []
        for g in groups:
            for l in g: hs.append(by_label[l]); ls.append(l)
            for _ in range(rows - len(g)): hs.append(spacer); ls.append("")
        leg = ax.legend(hs, ls, loc="upper left", bbox_to_anchor=(0.0, -0.24), fontsize=8, ncol=len(groups), frameon=False,
                        columnspacing=2.5, handletextpad=0.8, labelspacing=0.9)
        for h, txt in zip(getattr(leg, "legend_handles", None) or leg.legendHandles, leg.get_texts()):
            if "%" not in txt.get_text():
                try: h.set_sizes([30])
                except AttributeError: pass
        # card
        axc = fig.add_subplot(gs[0, 1:]); axc.axis("off")
        lines = self.card_lines(vid, full, pools, c, v)
        y = 1.0
        n_text = sum(1 for t, _ in lines if t); n_gap = len(lines) - n_text
        step = min(0.06, 1.0 / (n_text + 0.45 * n_gap))   # a long card closes up rather than running into the sections
        for text, bold in lines:
            if text: axc.text(0, y, text.replace("$", r"\$"), va="top", ha="left", family="monospace", fontsize=8.3, fontweight="bold" if bold else "normal", transform=axc.transAxes)
            y -= step if text else 0.45 * step
        # profiles
        tr_all = full.is_train.values == 1; eta_all = full.base.values + full.g_hat.values / self.height_scale
        s2_all = self.pool_variance(full.n.values[tr_all], full.k.values[tr_all], eta_all[tr_all]) if tr_all.sum() > 2 else 0.0
        s2_type = {}
        for is_multi in (False, True):
            m_ = tr_all & (full.multi.values == is_multi)
            s2_type[is_multi] = self.pool_variance(full.n.values[m_], full.k.values[m_], eta_all[m_]) if m_.sum() > 2 else s2_all
        ax1 = fig.add_subplot(gs[1, 1:]); self.profile(ax1, vid, pools, along, f"A ({end_types['A']})      along the parent segment, vertex at 0, parents at -1 and +1      A' ({end_types[chr(65) + chr(39)]})", color="#1f77b4", s2_type=s2_type, half=half, strip_unit=half_across)
        h_, l_ = ax1.get_legend_handles_labels()
        ax1.text(0.01, 0.97, "section A-A'", transform=ax1.transAxes, fontsize=9, color="#1f77b4", va="top")
        ax2 = fig.add_subplot(gs[2, 1:]); self.profile(ax2, vid, pools, perp, f"B ({end_types['B']})      across the parent segment, vertex at 0, +-1 at the edge of its column      B' ({end_types[chr(66) + chr(39)]})", color="#8e44ad", s2_type=s2_type, half=half_across, strip_unit=half)
        ax2.text(0.01, 0.97, "section B-B'", transform=ax2.transAxes, fontsize=9, color="#8e44ad", va="top")
        lo = min(ax1.get_ylim()[0], ax2.get_ylim()[0]); hi = max(ax1.get_ylim()[1], ax2.get_ylim()[1]); ax1.set_ylim(lo, hi); ax2.set_ylim(lo, hi)
        ax2.text(0.0, -0.27, "band: the bin's log-odds under the fit with one mean-preserving pool effect per pool, given an event; 16th to 84th percentile.\n"
                 "The effect's variance is fitted by marginal likelihood on the star's training pools, by issue type.", transform=ax2.transAxes, fontsize=6.5, color="0.35", va="top", linespacing=1.3)
        h2, l2 = ax2.get_legend_handles_labels(); seen = dict(zip(l_, h_)); seen.update({l: h for l, h in zip(l2, h2) if l not in seen})
        ax1.legend(list(seen.values()), list(seen.keys()), loc="lower left", bbox_to_anchor=(0.0, 1.02), fontsize=7.5, ncol=3, frameon=False, columnspacing=1.2)
        fig.savefig(out, bbox_inches="tight"); plt.close(fig)
        return c


    # ---- three dimensions ----------------------------------------------------------------
    AXIS_NAMES = ("loan age", "coupon", "average balance")
    AXIS_COLS = ("x", "y", "z")

    def axis_label(self, a, t):
        """A unit coordinate on axis a in its own units."""
        if a == 0: return f"{self.wala_range[0] + self.months(t):.1f}"
        if a == 1: return f"{self.wac_range[0] + self.points(t):.2f}"
        return f"\\${self.balance(t) / 1e3:,.0f}k"

    def slice_plane(self, ax, vid, faces, pools, share, show_held_out, plane, fixed):
        """One slice of the 3-D mesh through the vertex: the plane's two axes, the third held at the
        vertex's value. The cells the slice cuts (the column's shaded), the column's pools projected
        onto the plane (sized by their share of the information), the vertices lying in the slice."""
        v = self.v.loc[vid]; a, b = plane; value = self.at(v)[fixed]
        lo, hi = self.lo_cols, self.hi_cols
        def cut(frame):
            return frame[(frame[lo[fixed]] <= value + 1e-12) & (frame[hi[fixed]] >= value - 1e-12)]
        star = cut(faces)
        for _, f in star.iterrows():
            own = vid in [f[f"v{k}"] for k in self.CORNERS]
            ax.fill([f[lo[a]], f[hi[a]], f[hi[a]], f[lo[a]]], [f[lo[b]], f[lo[b]], f[hi[b]], f[hi[b]]], color="#e4e4e4" if own else "#f0f0f0", zorder=0)
        ra = (faces[lo[a]].min(), faces[hi[a]].max()); rb = (faces[lo[b]].min(), faces[hi[b]].max())
        pa, pb = 0.4 * (ra[1] - ra[0]) + 0.01, 0.4 * (rb[1] - rb[0]) + 0.01
        wa = (max(ra[0] - pa, 0), min(ra[1] + pa, 1)); wb = (max(rb[0] - pb, 0), min(rb[1] + pb, 1))
        win = cut(self.cells)
        win = win[(win[hi[a]] > wa[0]) & (win[lo[a]] < wa[1]) & (win[hi[b]] > wb[0]) & (win[lo[b]] < wb[1])]
        for _, f in win.iterrows():
            ax.plot([f[lo[a]], f[hi[a]], f[hi[a]], f[lo[a]], f[lo[a]]], [f[lo[b]], f[lo[b]], f[hi[b]], f[hi[b]], f[lo[b]]], color="#7f9db9", lw=0.5, zorder=1)
        P = self.coords(pools)
        size = np.where(pools.in_fit.values, 10 + 1600 * share, 36)
        order = np.argsort(-size)
        sc = ax.scatter(P[a][order], P[b][order], s=size[order], c=np.clip(pools.resid.values[order], -3, 3), cmap="RdBu_r", vmin=-3, vmax=3,
                        edgecolors="0.25", linewidths=0.4, zorder=3)
        # the vertices lying in the slice, and the parents' segment when it lies in it
        vc = [self.AXIS_COLS[a], self.AXIS_COLS[b], self.AXIS_COLS[fixed]]
        vs = self.v[(np.abs(self.v[vc[2]] - value) < 1e-9) & (self.v[vc[0]] >= wa[0]) & (self.v[vc[0]] <= wa[1]) & (self.v[vc[1]] >= wb[0]) & (self.v[vc[1]] <= wb[1])]
        free = vs.free_coefficient == 1
        ax.scatter(vs[vc[0]][~free], vs[vc[1]][~free], marker="D", s=26, c="#2e8b57", edgecolors="k", linewidths=0.5, zorder=4)
        ax.scatter(vs[vc[0]][free], vs[vc[1]][free], marker="D", s=36, c="#f1c40f", edgecolors="k", linewidths=0.6, zorder=5)
        ax.scatter([v[vc[0]]], [v[vc[1]]], marker="D", s=60, c="#f1c40f" if v.free_coefficient == 1 else "#2e8b57", edgecolors="k", linewidths=1.8, zorder=8)
        if int(v.parent0) >= 0:
            q0, q1 = self.v.loc[int(v.parent0)], self.v.loc[int(v.parent1)]
            if abs(q0[vc[2]] - value) < 1e-9 and abs(q1[vc[2]] - value) < 1e-9:
                ax.plot([q0[vc[0]], q1[vc[0]]], [q0[vc[1]], q1[vc[1]]], color="#1f77b4", lw=1.6, zorder=3.6)
        ax.set_xlim(*wa); ax.set_ylim(*wb)
        for setter, lim, axis in ((ax.set_xticks, wa, a), (ax.set_yticks, wb, b)):
            ticks = np.linspace(lim[0], lim[1], 5); setter(ticks)
            (ax.set_xticklabels if setter == ax.set_xticks else ax.set_yticklabels)([self.axis_label(axis, t) for t in ticks], fontsize=8)
        ax.set_xlabel(self.AXIS_NAMES[a] + (", months" if a == 0 else ", percent" if a == 1 else ""))
        ax.set_ylabel(self.AXIS_NAMES[b] + (", months" if b == 0 else ", percent" if b == 1 else ""))
        ax.set_title(f"slice at {self.AXIS_NAMES[fixed]} {self.axis_label(fixed, value)}: cells the slice cuts, the column's pools projected", fontsize=9)
        return sc

    def page3d(self, vid: int, out: Path, title: str = "", show_held_out: bool = False):
        """The page for a three-dimensional run: two slices through the vertex (loan age x coupon at
        its balance, loan age x balance at its coupon), the card, and the two sections."""
        full = self.star_pools(vid); c = self.card(vid, full); v = self.v.loc[vid]
        keep = np.ones(len(full), bool) if show_held_out else (full.is_train.values == 1)
        pools = full[keep]; share = c["share"][keep]
        faces = self.star_faces(vid)
        fig = plt.figure(figsize=(16, 14), dpi=130)
        gs = fig.add_gridspec(3, 3, width_ratios=[2.2, 1, 1], height_ratios=[2.3, 1, 1], wspace=0.28, hspace=0.5)
        sc = self.slice_plane(fig.add_subplot(gs[0:2, 0]), vid, faces, pools, share, show_held_out, (0, 1), 2)
        ax_b = fig.add_subplot(gs[2, 0])
        self.slice_plane(ax_b, vid, faces, pools, share, show_held_out, (0, 2), 1)
        cb = fig.colorbar(sc, ax=ax_b, orientation="horizontal", fraction=0.05, pad=0.25, aspect=40)
        cb.set_label("residual after the refit (red: more events than fitted); area: share of the vertex's information; yellow: free, green: constrained", fontsize=8)
        fig.suptitle((f"{title}   " if title else "") + f"vertex {self.address(v)} (id {vid}), {'free coefficient' if c['free'] else 'constrained vertex'}", fontsize=11, x=0.3, y=0.91)
        axc = fig.add_subplot(gs[0, 1:]); axc.axis("off")
        lines = self.card_lines(vid, full, pools, c, v)
        y = 1.0
        n_text = sum(1 for t, _ in lines if t); n_gap = len(lines) - n_text
        step = min(0.06, 1.0 / (n_text + 0.45 * n_gap))
        for text, bold in lines:
            if text: axc.text(0, y, text.replace("$", r"\$"), va="top", ha="left", family="monospace", fontsize=8.3, fontweight="bold" if bold else "normal", transform=axc.transAxes)
            y -= step if text else 0.45 * step
        # the sections: A along the parent segment (a root: along loan age), B across it (in the
        # loan age x coupon plane when the segment lies in it, else along loan age)
        axis = int(v.axis)
        along_axis = axis if axis >= 0 else 0
        across_axis = (1 if along_axis == 0 else 0) if axis in (0, 1) else (2 if axis < 0 else 0)
        e = np.eye(3)
        at = np.array(self.at(v))
        reach = [float(np.maximum(np.abs(faces[self.lo_cols[d]] - at[d]), np.abs(faces[self.hi_cols[d]] - at[d])).max()) for d in range(3)]
        if axis >= 0:
            q0, q1 = self.v.loc[int(v.parent0)], self.v.loc[int(v.parent1)]
            half = abs(self.at(q1)[axis] - self.at(q0)[axis]) / 2
        else:
            half = reach[along_axis]
        half_across = reach[across_axis]
        def end_kind(point):
            hit = self.v[(np.abs(self.v.x - point[0]) < 1e-6) & (np.abs(self.v.y - point[1]) < 1e-6) & (np.abs(self.v.z - point[2]) < 1e-6)]
            if len(hit) == 0: return "no vertex"
            return "free" if hit.free_coefficient.iloc[0] == 1 else "constrained"
        ends = {}
        for name, d, h in (("A", along_axis, half), ("B", across_axis, half_across)):
            for sign, lab in ((-1, name), (1, name + "'")):
                pt = at + sign * h * e[d]
                ends[lab] = "the cube's edge" if np.any(pt < -1e-12) or np.any(pt > 1 + 1e-12) else end_kind(np.clip(pt, 0, 1))
        tr_all = full.is_train.values == 1; eta_all = full.base.values + full.g_hat.values / self.height_scale
        s2_all = self.pool_variance(full.n.values[tr_all], full.k.values[tr_all], eta_all[tr_all]) if tr_all.sum() > 2 else 0.0
        s2_type = {}
        for is_multi in (False, True):
            m_ = tr_all & (full.multi.values == is_multi)
            s2_type[is_multi] = self.pool_variance(full.n.values[m_], full.k.values[m_], eta_all[m_]) if m_.sum() > 2 else s2_all
        what = "along the parent segment" if axis >= 0 else "along loan age"
        ax1 = fig.add_subplot(gs[1, 1:])
        self.profile(ax1, vid, pools, e[along_axis], f"A ({ends['A']})      {what} ({self.AXIS_NAMES[along_axis]}), vertex at 0, " + ("parents at -1 and +1" if axis >= 0 else "+-1 at the edge of its column") + f"      A' ({ends[chr(65) + chr(39)]})",
                     color="#1f77b4", s2_type=s2_type, half=half, strip_unit=max(half_across, 1e-9), reach=reach)
        ax1.text(0.01, 0.97, "section A-A'", transform=ax1.transAxes, fontsize=9, color="#1f77b4", va="top")
        ax2 = fig.add_subplot(gs[2, 1:])
        self.profile(ax2, vid, pools, e[across_axis], f"B ({ends['B']})      across ({self.AXIS_NAMES[across_axis]}), vertex at 0, +-1 at the edge of its column      B' ({ends[chr(66) + chr(39)]})",
                     color="#8e44ad", s2_type=s2_type, half=half_across, strip_unit=max(half, 1e-9), reach=reach)
        ax2.text(0.01, 0.97, "section B-B'", transform=ax2.transAxes, fontsize=9, color="#8e44ad", va="top")
        lo_, hi_ = min(ax1.get_ylim()[0], ax2.get_ylim()[0]), max(ax1.get_ylim()[1], ax2.get_ylim()[1]); ax1.set_ylim(lo_, hi_); ax2.set_ylim(lo_, hi_)
        ax2.text(0.0, -0.27, "sections: the column's pools, placed by their position along the line; "
                 "band: the bin's log-odds under the fit with one mean-preserving pool effect per pool, 16th to 84th percentile.",
                 transform=ax2.transAxes, fontsize=6.5, color="0.35", va="top", wrap=True)
        h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); seen = dict(zip(l1, h1)); seen.update({l: h for l, h in zip(l2, h2) if l not in seen})
        ax1.legend(list(seen.values()), list(seen.keys()), loc="lower left", bbox_to_anchor=(0.0, 1.02), fontsize=7.5, ncol=3, frameon=False, columnspacing=1.2)
        fig.savefig(out, bbox_inches="tight"); plt.close(fig)
        return c

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path); ap.add_argument("design", type=Path)
    ap.add_argument("--law", type=Path); ap.add_argument("--vertex", type=int)
    ap.add_argument("--artifact-months", type=float, default=0.0, help="hatch ages below this; 10 for the Ginnie cumulative proxy, 0 for a monthly snapshot")
    ap.add_argument("--held-out", action="store_true", help="also draw the held-out pools (open markers, residual on the ring) and include them in the section bins")
    ap.add_argument("--pick", choices=["giant", "keff", "surplus"])
    ap.add_argument("--out", type=Path, default=Path("star.png")); ap.add_argument("--title", default="")
    ap.add_argument("--compare", type=Path, help="another run directory fitted on the same design: its fit is drawn beside this one in the sections")
    ap.add_argument("--compare-label", help="what the compared run is, for the card (default: its directory name)")
    a = ap.parse_args(argv)
    run = Run(a.run_dir, a.design, a.law, a.artifact_months, compare=a.compare, compare_label=a.compare_label)
    vid = a.vertex
    if vid is None:
        t = run.rank()
        vid = int({"giant": t.sort_values("max_share").iloc[-1], "keff": t.sort_values("keff_share").iloc[-1],
                   "surplus": t.assign(a=t.surplus_logit.abs()).sort_values("a").iloc[-1]}[a.pick].id)
    c = (run.page3d if run.dim == 3 else run.page)(vid, a.out, a.title, show_held_out=a.held_out)
    print(f"vertex {vid}: pools {c['pools']}, K_eff {c['keff']:.1f}, max share {100 * c['max_share']:.0f}%, surplus {c['surplus_logit']:+.3f}, T {c['T']:.3f} S {c['S']:.3f} P {c['P']:.3f} R {c['R']:.3f} -> {a.out}")


if __name__ == "__main__":
    main()
