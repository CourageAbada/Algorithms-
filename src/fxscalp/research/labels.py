"""Candidate label specifications (Phase 2B-A). Specification + causal generation for DIAGNOSTICS only: no model, no PnL.

Notation (all times UTC ms, one SEGMENT at a time; a segment never contains a gap > SEGMENT_GAP_S, so nothing here can
bridge a feed hole, a daily break or a weekend):

  grid        1-second bars i = 0..n-1 with end time t_i = end_ms[i]; bar i covers [t_i - 1000, t_i).
  quotes      bid_c[i], ask_c[i] = prevailing quote at t_i (last tick <= t_i, carried through empty seconds);
              bid_h/bid_l/ask_h/ask_l[i] = extremes of the quote DURING bar i (empty bar -> the prevailing quote).
  observation t = t_i  (identical to the feature row timestamp; features use information <= t_i).
  latency     L in {0,1} seconds (scenario). Entry quote = quote at t_{i+L}.
  horizon     H seconds. The label window is bars i+L+1 .. i+L+H; the exit quote at timeout is quote at t_{i+L+H}.
  validity    a row is labelled only if i+L+H <= n-1 (the whole window lies inside the segment); otherwise status BOUNDARY.
              label_end_ms = t_i + (L+H)*1000 is the last instant of information the label uses (for purging).
  entry       LONG buys at ask_e = ask_c[i+L];  SHORT sells at bid_e = bid_c[i+L].  LONG exits at the bid, SHORT at the ask.
  extra cost  x = extra_rt_points * point (price units), from the cost scenario (commission + 2*slippage).

A. TRIPLE BARRIER (tb)
   sigma_i = std of 1-s log mid returns over the trailing W=600 s ending at t_i (NaN -> status WARMUP).
   B_i = max( m * sigma_i * mid_c[i] * sqrt(H) , k_spread * (ask_c[i]-bid_c[i]) ),  m = 1.0, k_spread = 3.0 (a priori).
   LONG  profit: first second j in window with bid_h[j] >= ask_e + B_i + x;  stop: bid_l[j] <= ask_e - B_i.
   SHORT profit: first second j with ask_l[j] <= bid_e - B_i - x;            stop: ask_h[j] >= bid_e + B_i.
   Outcome per side: +1 if profit occurs strictly before stop, -1 if stop occurs first OR in the same second (conservative),
   0 if neither (timeout).  Label: +1 (LONG) if LONG outcome is +1 and SHORT outcome is not +1 or LONG hit earlier;
   -1 (SHORT) symmetric; if both are +1 in the same second -> 0 and flagged ambiguous; otherwise 0 (NO_TRADE).
B. FIXED-HORIZON COST-ADJUSTED RETURN (fh)
   r_long  = (bid_c[i+L+H] - ask_c[i+L])/point - extra_rt_points ;  r_short = (bid_c[i+L] - ask_c[i+L+H])/point - extra_rt_points.
   LONG if r_long > margin (0 points) ; SHORT if r_short > margin ; else NO_TRADE. (r_long + r_short < 0 always, so at most
   one side can be positive.)  Continuous targets r_long, r_short (points) and bps = points*point/mid*1e4 are kept.
C. OPPORTUNITY / NO_TRADE (opp)
   net favourable excursion at second j: LONG  F_j = (max_{k<=j} bid_h[k] - ask_e)/point - extra_rt_points,
                                         SHORT F_j = (bid_e - min_{k<=j} ask_l[k])/point - extra_rt_points;
   adverse excursion A_j: LONG (ask_e - min_{k<=j} bid_l[k])/point, SHORT (max_{k<=j} ask_h[k] - bid_e)/point.
   Target T_i = kappa * ( (ask_c[i+L]-bid_c[i+L])/point + extra_rt_points ),  kappa = 2.0 (edge >= 2x the cost), rho = 1.0.
   Side is an OPPORTUNITY if some j in the window has F_j >= T_i and A_j <= F_j / rho. Label +1/-1/0 as above (earlier
   opportunity wins; same second -> 0, ambiguous). No barrier on volatility, no stop: it asks "was a cost-covering,
   not-too-adverse move available", independent of any exit rule.
All parameters above are fixed a priori from market structure (documented in docs/PHASE2B_RESEARCH_SPEC.md) and are NOT to be
tuned on label/model results.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from fxscalp.research.costs import CostScenario, resolve_scenario

LABEL_FAMILIES = ("tb", "fh", "opp")
CANDIDATE_HORIZONS_S = (15, 60, 300)
STATUS_OK, STATUS_WARMUP, STATUS_BOUNDARY = 0, 1, 2
CLASS_NAMES = {1: "LONG", -1: "SHORT", 0: "NO_TRADE"}
_BLOCK = 8192


@dataclass(frozen=True)
class LabelSpec:
    family: str                 # tb | fh | opp
    horizon_s: int
    vol_window_s: int = 600     # tb: trailing window for sigma
    barrier_m: float = 1.0      # tb: barrier = m * sigma * sqrt(H)
    barrier_spread_floor: float = 3.0   # tb: barrier >= k * spread
    fh_margin_points: float = 0.0       # fh: net edge required beyond cost
    opp_kappa: float = 2.0      # opp: target = kappa * cost
    opp_rho: float = 1.0        # opp: reward : adverse >= rho

    @property
    def name(self) -> str:
        return f"{self.family}_H{self.horizon_s}"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, **asdict(self)}


def candidate_specs() -> list[LabelSpec]:
    return [LabelSpec(f, h) for f in LABEL_FAMILIES for h in CANDIDATE_HORIZONS_S]


@dataclass(frozen=True)
class SegmentGrid:
    end_ms: np.ndarray
    bid_c: np.ndarray
    ask_c: np.ndarray
    bid_h: np.ndarray
    bid_l: np.ndarray
    ask_h: np.ndarray
    ask_l: np.ndarray
    point: float

    @property
    def n(self) -> int:
        return len(self.end_ms)

    @property
    def mid_c(self) -> np.ndarray:
        return (self.bid_c + self.ask_c) / 2.0


def grid_from_bars(bars: pd.DataFrame, point: float) -> SegmentGrid | None:
    """1-second bars of ONE segment (``features.bars.aggregate_bars`` output) -> SegmentGrid. Trailing incomplete bars are
    dropped; empty seconds carry the prevailing quote (never a future one)."""
    b = bars
    if len(b) == 0:
        return None
    keep = b["complete"].to_numpy(dtype=bool)
    if not keep.all():
        first_bad = int(np.argmin(keep))
        b = b.iloc[:first_bad]
    if len(b) == 0:
        return None
    ffill = lambda c: pd.Series(b[c].to_numpy()).ffill().to_numpy()  # noqa: E731
    bid_c, ask_c = ffill("bid_close"), ffill("ask_close")
    empty = b["empty"].to_numpy(dtype=bool)
    prev_bid = np.concatenate(([bid_c[0]], bid_c[:-1]))
    prev_ask = np.concatenate(([ask_c[0]], ask_c[:-1]))
    def path(col: str, prev: np.ndarray) -> np.ndarray:
        return np.where(empty, prev, b[col].to_numpy())
    return SegmentGrid(b["end_ms"].to_numpy(dtype="int64"), bid_c, ask_c, path("bid_high", prev_bid), path("bid_low", prev_bid),
                       path("ask_high", prev_ask), path("ask_low", prev_ask), float(point))


def trailing_sigma(mid_c: np.ndarray, window_s: int) -> np.ndarray:
    """Std of 1-s log mid returns over the trailing ``window_s`` returns ending at i (inclusive). NaN until a full window."""
    r = np.diff(np.log(mid_c), prepend=np.nan)
    s = pd.Series(r).rolling(window_s, min_periods=window_s).std(ddof=1).to_numpy()
    return s


def _first_hit(win: np.ndarray, cond_thr: np.ndarray, ge: bool) -> tuple[np.ndarray, np.ndarray]:
    m = (win >= cond_thr[:, None]) if ge else (win <= cond_thr[:, None])
    anyhit = m.any(axis=1)
    return np.where(anyhit, m.argmax(axis=1), 10**9), anyhit


def _windows(a: np.ndarray, lat: int, horizon: int, rows: int) -> np.ndarray:
    return sliding_window_view(a[lat + 1:], horizon)[:rows]


def _combine(win_long: np.ndarray, win_short: np.ndarray, t_long: np.ndarray, t_short: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Earlier success wins; same second -> NO_TRADE + ambiguous."""
    label = np.zeros(len(win_long), dtype="int8")
    amb = np.zeros(len(win_long), dtype=bool)
    both = win_long & win_short
    label[win_long & ~win_short] = 1
    label[win_short & ~win_long] = -1
    label[both & (t_long < t_short)] = 1
    label[both & (t_short < t_long)] = -1
    amb[both & (t_long == t_short)] = True
    return label, amb


def generate_labels(grid: SegmentGrid, spec: LabelSpec, scenario: CostScenario | str | None) -> dict[str, np.ndarray]:
    """Vectorised, causal label generation for one segment. Returns arrays of length ``grid.n`` (see module docstring)."""
    sc = resolve_scenario(scenario)
    n, H, L = grid.n, spec.horizon_s, sc.latency_s
    out = {"status": np.full(n, STATUS_BOUNDARY, dtype="int8"), "label": np.zeros(n, dtype="int8"),
           "label_end_ms": grid.end_ms + (L + H) * 1000, "ambiguous": np.zeros(n, dtype=bool)}
    rows = n - L - H                       # rows i = 0 .. rows-1 have a full window inside the segment
    if rows <= 0:
        return out
    x = sc.extra_rt_points * grid.point
    pt = grid.point
    ask_e, bid_e = grid.ask_c[L:L + rows], grid.bid_c[L:L + rows]
    fam = spec.family
    if fam == "fh":
        r_long = (grid.bid_c[L + H:L + H + rows] - ask_e) / pt - sc.extra_rt_points
        r_short = (bid_e - grid.ask_c[L + H:L + H + rows]) / pt - sc.extra_rt_points
        lab = np.zeros(rows, dtype="int8")
        lab[r_long > spec.fh_margin_points] = 1
        lab[r_short > spec.fh_margin_points] = -1
        out["r_long_points"] = np.full(n, np.nan)
        out["r_short_points"] = np.full(n, np.nan)
        out["r_long_points"][:rows], out["r_short_points"][:rows] = r_long, r_short
        out["label"][:rows] = lab
        out["status"][:rows] = STATUS_OK
        return out
    if fam == "tb":
        sigma = trailing_sigma(grid.mid_c, spec.vol_window_s)[:rows]
        spread = (grid.ask_c - grid.bid_c)[:rows]
        B = np.maximum(spec.barrier_m * sigma * grid.mid_c[:rows] * np.sqrt(H), spec.barrier_spread_floor * spread)
        valid = np.isfinite(sigma)
        out["barrier"] = np.full(n, np.nan)
        out["barrier"][:rows] = B
        for k in ("long_outcome", "short_outcome"):
            out[k] = np.zeros(n, dtype="int8")
        out["t_hit"] = np.full(n, -1, dtype="int32")
        lab_all = np.zeros(rows, dtype="int8")
        amb_all = np.zeros(rows, dtype=bool)
        lo_all = np.zeros(rows, dtype="int8")
        so_all = np.zeros(rows, dtype="int8")
        thit = np.full(rows, -1, dtype="int32")
        wbh, wbl = _windows(grid.bid_h, L, H, rows), _windows(grid.bid_l, L, H, rows)
        wal, wah = _windows(grid.ask_l, L, H, rows), _windows(grid.ask_h, L, H, rows)
        for s in range(0, rows, _BLOCK):
            e = min(rows, s + _BLOCK)
            sl = slice(s, e)
            Bk = B[sl]
            lp, lp_any = _first_hit(wbh[sl], ask_e[sl] + Bk + x, True)
            ls, ls_any = _first_hit(wbl[sl], ask_e[sl] - Bk, False)
            sp_, sp_any = _first_hit(wal[sl], bid_e[sl] - Bk - x, False)
            ss, ss_any = _first_hit(wah[sl], bid_e[sl] + Bk, True)
            l_win = lp_any & (~ls_any | (lp < ls))
            s_win = sp_any & (~ss_any | (sp_ < ss))
            lab, amb = _combine(l_win, s_win, lp, sp_)
            lo = np.where(l_win, 1, np.where(ls_any, -1, 0)).astype("int8")
            so = np.where(s_win, 1, np.where(ss_any, -1, 0)).astype("int8")
            lab_all[sl], amb_all[sl], lo_all[sl], so_all[sl] = lab, amb, lo, so
            t = np.where(lab == 1, lp, np.where(lab == -1, sp_, -1))
            thit[sl] = np.where(lab != 0, t + 1, -1)
        lab_all[~valid], amb_all[~valid] = 0, False
        out["label"][:rows], out["ambiguous"][:rows] = lab_all, amb_all
        out["long_outcome"][:rows], out["short_outcome"][:rows], out["t_hit"][:rows] = lo_all, so_all, thit
        st = np.where(valid, STATUS_OK, STATUS_WARMUP).astype("int8")
        out["status"][:rows] = st
        return out
    if fam == "opp":
        cost = (grid.ask_c - grid.bid_c)[L:L + rows] / pt + sc.extra_rt_points
        T = spec.opp_kappa * cost
        wbh, wbl = _windows(grid.bid_h, L, H, rows), _windows(grid.bid_l, L, H, rows)
        wal, wah = _windows(grid.ask_l, L, H, rows), _windows(grid.ask_h, L, H, rows)
        lab_all = np.zeros(rows, dtype="int8")
        amb_all = np.zeros(rows, dtype=bool)
        for s in range(0, rows, _BLOCK):
            e = min(rows, s + _BLOCK)
            sl = slice(s, e)
            fav_l = (np.maximum.accumulate(wbh[sl], axis=1) - ask_e[sl, None]) / pt - sc.extra_rt_points
            adv_l = (ask_e[sl, None] - np.minimum.accumulate(wbl[sl], axis=1)) / pt
            fav_s = (bid_e[sl, None] - np.minimum.accumulate(wal[sl], axis=1)) / pt - sc.extra_rt_points
            adv_s = (np.maximum.accumulate(wah[sl], axis=1) - bid_e[sl, None]) / pt
            ok_l = (fav_l >= T[sl, None]) & (adv_l <= fav_l / spec.opp_rho)
            ok_s = (fav_s >= T[sl, None]) & (adv_s <= fav_s / spec.opp_rho)
            tl = np.where(ok_l.any(axis=1), ok_l.argmax(axis=1), 10**9)
            ts_ = np.where(ok_s.any(axis=1), ok_s.argmax(axis=1), 10**9)
            lab_all[sl], amb_all[sl] = _combine(ok_l.any(axis=1), ok_s.any(axis=1), tl, ts_)
        out["label"][:rows], out["ambiguous"][:rows] = lab_all, amb_all
        out["status"][:rows] = STATUS_OK
        return out
    raise ValueError(f"unknown label family {fam!r}")


# ------------------------------------------------------------------------------------------------ reference implementation
def reference_labels(grid: SegmentGrid, spec: LabelSpec, scenario: CostScenario | str | None) -> dict[str, np.ndarray]:
    """Slow, loop-based implementation of the SAME definitions, used by the tests as an independent oracle."""
    sc = resolve_scenario(scenario)
    n, H, L, pt = grid.n, spec.horizon_s, sc.latency_s, grid.point
    x = sc.extra_rt_points * pt
    label = np.zeros(n, dtype="int8")
    status = np.full(n, STATUS_BOUNDARY, dtype="int8")
    sigma = trailing_sigma(grid.mid_c, spec.vol_window_s)
    for i in range(n):
        if i + L + H > n - 1:
            continue
        status[i] = STATUS_OK
        ae, be = grid.ask_c[i + L], grid.bid_c[i + L]
        if spec.family == "fh":
            rl = (grid.bid_c[i + L + H] - ae) / pt - sc.extra_rt_points
            rs = (be - grid.ask_c[i + L + H]) / pt - sc.extra_rt_points
            label[i] = 1 if rl > spec.fh_margin_points else (-1 if rs > spec.fh_margin_points else 0)
            continue
        js = range(i + L + 1, i + L + H + 1)
        if spec.family == "tb":
            if not np.isfinite(sigma[i]):
                status[i] = STATUS_WARMUP
                continue
            B = max(spec.barrier_m * sigma[i] * grid.mid_c[i] * np.sqrt(H), spec.barrier_spread_floor * (grid.ask_c[i] - grid.bid_c[i]))
            def first(pred):
                for k, j in enumerate(js):
                    if pred(j):
                        return k
                return None
            lp = first(lambda j: grid.bid_h[j] >= ae + B + x)
            ls = first(lambda j: grid.bid_l[j] <= ae - B)
            sp = first(lambda j: grid.ask_l[j] <= be - B - x)
            ss = first(lambda j: grid.ask_h[j] >= be + B)
            l_win = lp is not None and (ls is None or lp < ls)
            s_win = sp is not None and (ss is None or sp < ss)
        else:   # opp
            T = spec.opp_kappa * ((ae - be) / pt + sc.extra_rt_points)
            mx_b = mn_b = None
            lp = sp = None
            hi_b, lo_b, lo_a, hi_a = -np.inf, np.inf, np.inf, -np.inf
            for k, j in enumerate(js):
                hi_b, lo_b = max(hi_b, grid.bid_h[j]), min(lo_b, grid.bid_l[j])
                lo_a, hi_a = min(lo_a, grid.ask_l[j]), max(hi_a, grid.ask_h[j])
                fl, al = (hi_b - ae) / pt - sc.extra_rt_points, (ae - lo_b) / pt
                fs, as_ = (be - lo_a) / pt - sc.extra_rt_points, (hi_a - be) / pt
                if lp is None and fl >= T and al <= fl / spec.opp_rho:
                    lp = k
                if sp is None and fs >= T and as_ <= fs / spec.opp_rho:
                    sp = k
            l_win, s_win = lp is not None, sp is not None
        if l_win and s_win:
            label[i] = 1 if lp < sp else (-1 if sp < lp else 0)
        elif l_win:
            label[i] = 1
        elif s_win:
            label[i] = -1
    return {"status": status, "label": label}
