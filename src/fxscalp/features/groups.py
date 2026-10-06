"""Feature groups. Every feature is emitted together with its metadata (definition, units, lookback, NaN behaviour,
leakage assessment, reason), so the registry can never drift from what is computed.

RULES (enforced by tests/leakage and ``leakage.scan_source``): group code may use only ``ops`` primitives, the context
accessors and element-wise arithmetic. No direct statistics, no negative shifts, no centred windows, no reversed arrays.

Conventions
-----------
* Row t = a decision at the END of grid bar t; it may use only ticks with timestamp < t.
* Price distances/returns are in basis points (1e-4 of price) so they are unit-free across price levels.
* Spread in points = spread / point, with ``point`` taken from stored symbol metadata (NaN if unknown).
* "as-of" state (last known price/spread) is explicit and always paired with ``staleness_s``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from fxscalp.features import ops
from fxscalp.features.bars import align_completed, take
from fxscalp.features.session_features import session_arrays
from fxscalp.features.spec import FeatureConfig, FeatureSpec
from fxscalp.regimes.config import RegimeConfig
from fxscalp.sessions.calendar import DEFAULT_RULES, SessionRules

LEAK_BACK = "LOW: trailing window over past rows only (ops primitives; prefix-invariance tested)"
LEAK_MTF = ("MEDIUM: uses a higher-timeframe bar only after it is complete (align_completed side='right'); "
            "covered by the MTF regression test and prefix-invariance")
LEAK_SESSION = ("MEDIUM: running aggregate inside the current session instance (never the final session value); "
                "covered by the session-so-far tests and prefix-invariance")
LEAK_STATE = "LOW: as-of state of the last tick before the decision time"
NAN_WARM = "NaN during warm-up (first lookback_s of each segment) and where the denominator is zero / no ticks"


@dataclass
class FeatureContext:
    """Everything a group may read, for ONE continuous segment of ticks on a uniform grid."""

    cfg: FeatureConfig
    grid_s: int
    point: float | None
    base: pd.DataFrame                       # complete base bars (rows == decision rows)
    mtf: dict[int, pd.DataFrame]             # timeframe_s -> bars (incl. incomplete; alignment filters)
    rules: SessionRules = DEFAULT_RULES
    regime_cfg: RegimeConfig = field(default_factory=RegimeConfig)
    _cache: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        b = self.base
        self.start_ms = b["start_ms"].to_numpy()
        self.end_ms = b["end_ms"].to_numpy()
        self.n = len(b)
        self.n_ticks = b["n_ticks"].to_numpy(dtype="float64")
        self.mid_asof = ops.ffill_asof(b["mid_close"].to_numpy())
        self.bid_asof = ops.ffill_asof(b["bid_close"].to_numpy())
        self.ask_asof = ops.ffill_asof(b["ask_close"].to_numpy())
        self.spread_asof = ops.ffill_asof(b["spread_close"].to_numpy())
        self.last_tick_asof = ops.ffill_asof(b["last_tick_ms"].to_numpy().astype("float64"))

    def k(self, w_s: int) -> int:
        if w_s % self.grid_s:
            raise ValueError(f"window {w_s}s is not a multiple of the grid ({self.grid_s}s)")
        return max(1, w_s // self.grid_s)

    def col(self, name: str) -> np.ndarray:
        return self.base[name].to_numpy(dtype="float64")

    def completed(self, tf_s: int) -> tuple[np.ndarray, np.ndarray]:
        if tf_s not in self._cache:
            self._cache[tf_s] = align_completed(self.end_ms, self.mtf[tf_s])
        return self._cache[tf_s]

    def pts(self, x: np.ndarray) -> np.ndarray:
        return x / self.point if self.point and self.point > 0 else np.full(np.shape(x), np.nan)


class Emitter:
    def __init__(self, group: str, version: int, disabled: tuple[str, ...]):
        self.group, self.version, self.disabled = group, version, set(disabled)
        self.items: list[tuple[FeatureSpec, np.ndarray]] = []

    def __call__(self, name: str, arr: np.ndarray, *, family: str, desc: str, source: str, tf: str, lookback: int,
                 units: str, rng: tuple[float | None, float | None] | None, reason: str, nan: str = NAN_WARM,
                 leak: str = LEAK_BACK) -> None:
        spec = FeatureSpec(name, self.group, family, desc, source, tf, int(lookback), units, rng, nan, leak, reason,
                           enabled=name not in self.disabled, version=self.version)
        self.items.append((spec, np.asarray(arr, dtype="float64")))


GroupFn = Callable[[FeatureContext, FeatureConfig, dict, Emitter], None]


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    definition_version: int
    fn: GroupFn


# ----------------------------------------------------------------------------------------------------------------
def micro(ctx: FeatureContext, cfg: FeatureConfig, prior: dict, emit: Emitter) -> None:
    g = f"{ctx.grid_s}s"
    nt, up, dn, fl = ctx.n_ticks, ctx.col("n_up"), ctx.col("n_down"), ctx.col("n_flat")
    same, opp, sdt, ndt = ctx.col("n_same"), ctx.col("n_opp"), ctx.col("sum_dt"), ctx.col("n_dt")
    for w in cfg.micro_windows_s:
        k = ctx.k(w)
        cnt = ops.rsum(nt, k)
        rate = cnt / w
        emit(f"tick_count_{w}s", cnt, family="microstructure", desc=f"Ticks received in the last {w}s", source="ticks",
             tf=g, lookback=w, units="ticks", rng=(0, None), reason="Activity level / information arrival")
        emit(f"tick_rate_{w}s", rate, family="microstructure", desc=f"tick_count_{w}s divided by {w}", source="ticks",
             tf=g, lookback=w, units="ticks/s", rng=(0, None), reason="Tick arrival rate")
        emit(f"tick_acceleration_{w}s", (rate - ops.lag(rate, k)) / w, family="microstructure",
             desc=f"Change of tick rate vs the previous {w}s window, per {w}s", source="ticks", tf=g, lookback=2 * w,
             units="ticks/s^2", rng=None, reason="Burst onset / fade-out")
        emit(f"tick_interval_mean_ms_{w}s", ops.ratio(ops.rsum(sdt, k), ops.rsum(ndt, k)), family="microstructure",
             desc=f"Mean inter-tick interval (ms) over the last {w}s; the interval is attributed to the later tick",
             source="ticks", tf=g, lookback=w, units="ms", rng=(0, None), reason="Arrival interval (inverse of rate, "
             "robust to empty bars)")
        moves = ops.rsum(up, k) + ops.rsum(dn, k) + ops.rsum(fl, k)
        u, d = ops.rsum(up, k), ops.rsum(dn, k)
        emit(f"up_tick_ratio_{w}s", ops.ratio(u, moves), family="microstructure",
             desc=f"Fraction of ticks in the last {w}s whose mid rose vs the previous tick", source="ticks", tf=g,
             lookback=w, units="ratio", rng=(0, 1), reason="Directional pressure")
        emit(f"down_tick_ratio_{w}s", ops.ratio(d, moves), family="microstructure",
             desc=f"Fraction of ticks in the last {w}s whose mid fell vs the previous tick", source="ticks", tf=g,
             lookback=w, units="ratio", rng=(0, 1), reason="Directional pressure")
        emit(f"tick_direction_imbalance_{w}s", ops.ratio(u - d, u + d), family="microstructure",
             desc=f"(up - down) / (up + down) mid moves in the last {w}s", source="ticks", tf=g, lookback=w,
             units="ratio", rng=(-1, 1), reason="Signed directional imbalance")
        emit(f"directional_persistence_{w}s", ops.ratio(ops.rsum(same, k), ops.rsum(same, k) + ops.rsum(opp, k)),
             family="microstructure", desc=f"Share of consecutive non-flat moves that continue the same direction "
             f"({w}s)", source="ticks", tf=g, lookback=w, units="ratio", rng=(0, 1), reason="Trending vs bouncing quotes")
    run = ops.ffill_asof(ctx.col("run_close"))
    emit("consecutive_up_ticks", np.maximum(run, 0), family="microstructure",
         desc="Length of the current run of consecutive up moves (flat ticks ignored), as of the last tick",
         source="ticks", tf=g, lookback=0, units="ticks", rng=(0, None), nan="NaN only before the first tick",
         leak=LEAK_STATE, reason="Short-term directional persistence")
    emit("consecutive_down_ticks", np.maximum(-run, 0), family="microstructure",
         desc="Length of the current run of consecutive down moves, as of the last tick", source="ticks", tf=g,
         lookback=0, units="ticks", rng=(0, None), nan="NaN only before the first tick", leak=LEAK_STATE,
         reason="Short-term directional persistence")
    emit("staleness_s", (ctx.end_ms - ctx.last_tick_asof) / 1000.0, family="state",
         desc="Seconds between the decision time and the last tick (age of every as-of value)", source="ticks", tf=g,
         lookback=0, units="s", rng=(0, None), nan="never NaN", leak=LEAK_STATE,
         reason="Makes stale as-of state explicit; quality gate")
    emit("empty_interval", (nt == 0).astype("float64"), family="state",
         desc="1 if the last grid interval contained no tick", source="ticks", tf=g, lookback=0, units="flag",
         rng=(0, 1), nan="never NaN", leak=LEAK_BACK, reason="Explicit marker for empty intervals")


def spread(ctx: FeatureContext, cfg: FeatureConfig, prior: dict, emit: Emitter) -> None:
    g = f"{ctx.grid_s}s"
    sp_pts = ctx.pts(ctx.spread_asof)
    emit("spread_points", sp_pts, family="spread", desc="Spread (ask-bid) of the last tick in points (spread/point)",
         source="ticks", tf=g, lookback=0, units="points", rng=(0, None), nan="NaN if symbol point is unknown",
         leak=LEAK_STATE, reason="Transaction-cost proxy; unit comes from stored symbol metadata")
    emit("relative_spread_bps", ops.bps(ctx.spread_asof, ctx.mid_asof), family="spread",
         desc="Spread of the last tick relative to mid", source="ticks", tf=g, lookback=0, units="bps", rng=(0, None),
         nan="never NaN after the first tick", leak=LEAK_STATE, reason="Unit-free cost measure")
    for w in cfg.spread_change_windows_s:
        emit(f"spread_change_points_{w}s", sp_pts - ops.lag(sp_pts, ctx.k(w)), family="spread",
             desc=f"Spread now minus spread {w}s ago (points)", source="ticks", tf=g, lookback=w, units="points",
             rng=None, reason="Liquidity withdrawal / return")
    nt = ctx.n_ticks
    for w in cfg.spread_mean_windows_s:
        k = ctx.k(w)
        emit(f"spread_mean_points_{w}s", ctx.pts(ops.ratio(ops.rsum(ctx.col("spread_sum"), k), ops.rsum(nt, k))),
             family="spread", desc=f"Tick-weighted mean spread over the last {w}s (points)", source="ticks", tf=g,
             lookback=w, units="points", rng=(0, None), reason="Smoothed cost level")
    for w in cfg.spread_median_windows_s:
        emit(f"spread_median_points_{w}s", ctx.pts(ops.rmedian(ctx.col("spread_median"), ctx.k(w), cfg.min_median_frac)),
             family="spread", desc=f"Median over the last {w}s of the per-bar median spread (non-empty bars only; "
             "an approximation of the tick median)", source="ticks", tf=g, lookback=w, units="points", rng=(0, None),
             reason="Outlier-robust cost level")
    kh = ctx.k(cfg.spread_hist_s)
    emit(f"spread_percentile_{cfg.spread_hist_s}s", ops.rrank(sp_pts, kh), family="spread",
         desc=f"Percentile rank of the current spread within the trailing {cfg.spread_hist_s}s of per-bar as-of "
         "spreads", source="ticks", tf=g, lookback=cfg.spread_hist_s, units="ratio", rng=(0, 1),
         reason="Relative spread level (distribution-based, no fixed threshold)")
    mu, sd = ops.rmean(sp_pts, kh), ops.rstd(sp_pts, kh)
    emit(f"spread_zscore_{cfg.spread_hist_s}s", ops.ratio(sp_pts - mu, sd), family="spread",
         desc=f"(spread - trailing mean) / trailing std over {cfg.spread_hist_s}s", source="ticks", tf=g,
         lookback=cfg.spread_hist_s, units="z", rng=None, reason="Abnormal widening detection")


def structure(ctx: FeatureContext, cfg: FeatureConfig, prior: dict, emit: Emitter) -> None:
    g = f"{ctx.grid_s}s"
    mid, hi_b, lo_b = ctx.mid_asof, ctx.col("mid_high"), ctx.col("mid_low")
    hi, lo, width = {}, {}, {}
    for w in cfg.range_windows_s:
        k = ctx.k(w)
        hi[w], lo[w] = ops.rmax(hi_b, k), ops.rmin(lo_b, k)
        width[w] = ops.bps(hi[w] - lo[w], mid)
        emit(f"range_width_bps_{w}s", width[w], family="range", desc=f"(rolling high - rolling low) / mid over {w}s",
             source="ticks", tf=g, lookback=w, units="bps", rng=(0, None), reason="Local range / volatility proxy")
        emit(f"range_position_{w}s", ops.ratio(mid - lo[w], hi[w] - lo[w]), family="range",
             desc=f"Position of the last mid inside the {w}s range (0=low, 1=high)", source="ticks", tf=g, lookback=w,
             units="ratio", rng=(0, 1), reason="Mean-reversion vs breakout context")
        emit(f"dist_high_bps_{w}s", ops.bps(hi[w] - mid, mid), family="range",
             desc=f"Distance from the last mid up to the {w}s rolling high", source="ticks", tf=g, lookback=w,
             units="bps", rng=(0, None), reason="Normalised distance from rolling high")
        emit(f"dist_low_bps_{w}s", ops.bps(mid - lo[w], mid), family="range",
             desc=f"Distance from the {w}s rolling low up to the last mid", source="ticks", tf=g, lookback=w,
             units="bps", rng=(0, None), reason="Normalised distance from rolling low")
    ws = sorted(cfg.range_windows_s)
    short, long_ = ws[0], ws[-1]
    emit(f"compression_ratio_{short}s_{long_}s", ops.ratio(width[short], width[long_]), family="range",
         desc=f"range_width_{short}s / range_width_{long_}s (small = compression)", source="ticks", tf=g, lookback=long_,
         units="ratio", rng=(0, 1), reason="Volatility compression")
    mid_w = 60 if 60 in width else ws[len(ws) // 2]
    typical = ops.rmedian(width[mid_w], ctx.k(cfg.spread_hist_s), cfg.min_median_frac)
    emit(f"range_expansion_ratio_{mid_w}s_vs_{cfg.spread_hist_s // 60}m", ops.ratio(width[mid_w], typical),
         family="range", desc=f"range_width_{mid_w}s relative to its trailing median over {cfg.spread_hist_s}s",
         source="ticks", tf=g, lookback=mid_w + cfg.spread_hist_s, units="ratio", rng=(0, None),
         reason="Volatility expansion")
    emit(f"range_percentile_{mid_w}s_{cfg.vol_hist_s // 3600}h", ops.rrank(width[mid_w], ctx.k(cfg.vol_hist_s)),
         family="range", desc=f"Percentile rank of range_width_{mid_w}s within the trailing {cfg.vol_hist_s}s",
         source="ticks", tf=g, lookback=mid_w + cfg.vol_hist_s, units="ratio", rng=(0, 1),
         reason="Relative range level")
    ks = ctx.k(cfg.breakout_guard_s)
    for w in cfg.breakout_windows_s:
        k = ctx.k(w)
        ph, pl = ops.lag(ops.rmax(hi_b, k), ks), ops.lag(ops.rmin(lo_b, k), ks)
        emit(f"breakout_dist_up_bps_{w}s", ops.bps(mid - ph, mid), family="range",
             desc=f"Last mid minus the prior {w}s high (window ends {cfg.breakout_guard_s}s ago); >0 = above it",
             source="ticks", tf=g, lookback=w + cfg.breakout_guard_s, units="bps", rng=None,
             reason="Breakout magnitude (signed)")
        emit(f"breakout_dist_down_bps_{w}s", ops.bps(pl - mid, mid), family="range",
             desc=f"Prior {w}s low minus the last mid; >0 = below it", source="ticks", tf=g,
             lookback=w + cfg.breakout_guard_s, units="bps", rng=None, reason="Breakout magnitude (signed)")
    wf, lf = cfg.failed_breakout_window_s, cfg.failed_breakout_lookback_s
    kf = ctx.k(lf)
    lvl_h, lvl_l = ops.lag(ops.rmax(hi_b, ctx.k(wf)), kf), ops.lag(ops.rmin(lo_b, ctx.k(wf)), kf)
    rec_h, rec_l = ops.rmax(hi_b, kf), ops.rmin(lo_b, kf)
    ok_h, ok_l = ~np.isnan(lvl_h) & ~np.isnan(rec_h), ~np.isnan(lvl_l) & ~np.isnan(rec_l)
    emit(f"failed_breakout_up_{wf}s_{lf}s", np.where(ok_h, ((rec_h > lvl_h) & (mid < lvl_h)).astype("float64"), np.nan),
         family="range", desc=f"1 if price exceeded the {wf}s high (measured {lf}s ago) within the last {lf}s but is "
         "now back below it", source="ticks", tf=g, lookback=wf + lf, units="flag", rng=(0, 1),
         reason="Failed-breakout state (uses only past ticks)")
    emit(f"failed_breakout_down_{wf}s_{lf}s", np.where(ok_l, ((rec_l < lvl_l) & (mid > lvl_l)).astype("float64"), np.nan),
         family="range", desc=f"1 if price broke below the {wf}s low (measured {lf}s ago) within the last {lf}s but is "
         "now back above it", source="ticks", tf=g, lookback=wf + lf, units="flag", rng=(0, 1),
         reason="Failed-breakout state (uses only past ticks)")


def volatility(ctx: FeatureContext, cfg: FeatureConfig, prior: dict, emit: Emitter) -> None:
    g = f"{ctx.grid_s}s"
    sq = ctx.col("sum_sq_ret")
    rv = {}
    for w in cfg.vol_windows_s:
        rv[w] = np.sqrt(ops.rsum(sq, ctx.k(w))) * 1e4
        emit(f"realized_vol_bps_{w}s", rv[w], family="volatility",
             desc=f"sqrt(sum of squared tick mid log-returns) over {w}s (includes microstructure noise)",
             source="ticks", tf=g, lookback=w, units="bps", rng=(0, None), reason="Short-term realised volatility")
    for tf in cfg.atr_timeframes_s:
        bars = ctx.mtf[tf]
        hi, lo, cl = (bars[c].to_numpy(dtype="float64") for c in ("mid_high", "mid_low", "mid_close"))
        prev = ops.lag(ops.ffill_asof(cl), 1)
        tr = np.fmax(np.fmax(hi - lo, np.abs(hi - prev)), np.abs(lo - prev))
        atr = ops.rmean(tr, cfg.atr_period)
        atr_bps = ops.bps(atr, ops.ffill_asof(cl))
        idx, _ = ctx.completed(tf)
        emit(f"atr{cfg.atr_period}_bps_{tf}s", take(atr_bps, idx), family="volatility",
             desc=f"Average true range ({cfg.atr_period} bars, mid) on COMPLETE {tf}s bars, relative to close",
             source=f"bars_{tf}s", tf=f"{tf}s", lookback=tf * (cfg.atr_period + 1), units="bps", rng=(0, None),
             leak=LEAK_MTF, reason="Tick-equivalent ATR for volatility scaling")
    w60 = 60 if 60 in rv else sorted(rv)[len(rv) // 2]
    kh = ctx.k(cfg.vol_hist_s)
    emit(f"vol_percentile_{w60}s_{cfg.vol_hist_s // 3600}h", ops.rrank(rv[w60], kh), family="volatility",
         desc=f"Percentile rank of realized_vol_{w60}s within the trailing {cfg.vol_hist_s}s", source="ticks", tf=g,
         lookback=w60 + cfg.vol_hist_s, units="ratio", rng=(0, 1), reason="Relative volatility level")
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = np.where(rv[w60] > 0, np.log(rv[w60]), np.nan)
    emit(f"vol_zscore_{w60}s_{cfg.vol_hist_s // 3600}h", ops.ratio(lg - ops.rmean(lg, kh), ops.rstd(lg, kh)),
         family="volatility", desc=f"z-score of ln(realized_vol_{w60}s) vs its trailing {cfg.vol_hist_s}s mean/std",
         source="ticks", tf=g, lookback=w60 + cfg.vol_hist_s, units="z", rng=None, reason="Volatility regime scaling")


def momentum(ctx: FeatureContext, cfg: FeatureConfig, prior: dict, emit: Emitter) -> None:
    g = f"{ctx.grid_s}s"
    mid = ctx.mid_asof
    rets = {}
    for w in cfg.ret_windows_s:
        rets[w] = ops.log_ret_bps(mid, ctx.k(w))
        emit(f"ret_bps_{w}s", rets[w], family="momentum", desc=f"ln(mid_t / mid_(t-{w}s)) using as-of mids",
             source="ticks", tf=g, lookback=w, units="bps", rng=None, reason="Multi-horizon momentum / rolling return")
    for w in cfg.abs_ret_windows_s:
        emit(f"abs_ret_bps_{w}s", np.abs(rets[w]), family="momentum", desc=f"|ret_bps_{w}s|", source="ticks", tf=g,
             lookback=w, units="bps", rng=(0, None), reason="Magnitude of recent movement")
    acc = np.zeros(ctx.n)
    for w in cfg.ret_windows_s:
        acc = acc + np.sign(rets[w])
    emit("momentum_alignment", acc / len(cfg.ret_windows_s), family="momentum",
         desc="Mean sign of ret_bps over all horizons (+1 = all up, -1 = all down)", source="ticks", tf=g,
         lookback=sorted(cfg.ret_windows_s)[-1], units="ratio", rng=(-1, 1), reason="Agreement across horizons")
    ks = ctx.k(cfg.ema_slope_step_s)
    for span in cfg.ema_spans_s:
        e = ops.ewma(mid, ctx.k(span))
        emit(f"ema_slope_bps_per_s_{span}s", ops.bps(e - ops.lag(e, ks), e) / (ks * ctx.grid_s), family="momentum",
             desc=f"Slope of the {span}s EMA of mid over the last {cfg.ema_slope_step_s}s", source="ticks", tf=g,
             lookback=span + cfg.ema_slope_step_s, units="bps/s", rng=None, reason="Trend direction")
        emit(f"ema_dist_bps_{span}s", ops.bps(mid - e, mid), family="momentum",
             desc=f"(mid - EMA{span}s) / mid; also serves as the mean-reversion distance from EMA", source="ticks",
             tf=g, lookback=span, units="bps", rng=None, reason="Normalised EMA distance (trend / reversion)")
    w = 15 if 15 in rets else sorted(rets)[0]
    vel = rets[w] / w
    emit(f"price_velocity_bps_per_s_{w}s", vel, family="momentum", desc=f"ret_bps_{w}s / {w}", source="ticks", tf=g,
         lookback=w, units="bps/s", rng=None, reason="Price velocity")
    emit(f"price_acceleration_bps_per_s2_{w}s", (vel - ops.lag(vel, ctx.k(w))) / w, family="momentum",
         desc=f"Change of price velocity vs {w}s earlier, per {w}s", source="ticks", tf=g, lookback=2 * w,
         units="bps/s^2", rng=None, reason="Return acceleration")


def mean_reversion(ctx: FeatureContext, cfg: FeatureConfig, prior: dict, emit: Emitter) -> None:
    g = f"{ctx.grid_s}s"
    mid, nt = ctx.mid_asof, ctx.n_ticks
    for w in cfg.mean_windows_s:
        k = ctx.k(w)
        mu, sd = ops.rmean(mid, k), ops.rstd(mid, k)
        emit(f"dist_mean_bps_{w}s", ops.bps(mid - mu, mid), family="mean_reversion",
             desc=f"(mid - mean of as-of mids over {w}s) / mid", source="ticks", tf=g, lookback=w, units="bps",
             rng=None, reason="Distance from rolling mean")
        emit(f"zscore_mean_{w}s", ops.ratio(mid - mu, sd), family="mean_reversion",
             desc=f"(mid - rolling mean) / rolling std over {w}s", source="ticks", tf=g, lookback=w, units="z",
             rng=None, reason="Standardised stretch")
        tw = ops.ratio(ops.rsum(ctx.col("mid_sum"), k), ops.rsum(nt, k))
        emit(f"dist_tickweighted_mean_bps_{w}s", ops.bps(mid - tw, mid), family="mean_reversion",
             desc=f"(mid - tick-count-weighted mean mid over {w}s) / mid. NOT a VWAP: broker tick volume is an update "
             "count, not traded volume", source="ticks", tf=g, lookback=w, units="bps", rng=None,
             reason="Activity-weighted reference level (VWAP substitute, limitations documented)")


def mtf(ctx: FeatureContext, cfg: FeatureConfig, prior: dict, emit: Emitter) -> None:
    mid = ctx.mid_asof
    for tf in cfg.mtf_timeframes_s:
        bars = ctx.mtf[tf]
        idx, _ = ctx.completed(tf)
        o, h, l_, c = (take(bars[x].to_numpy(dtype="float64"), idx) for x in ("mid_open", "mid_high", "mid_low", "mid_close"))
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where((o > 0) & (c > 0), 1e4 * np.log(c / o), np.nan)
        kw = dict(family="mtf", source=f"bars_{tf}s", tf=f"{tf}s", lookback=2 * tf, leak=LEAK_MTF)
        emit(f"htf_ret_bps_{tf}s", r, desc=f"Open-to-close return of the last COMPLETE {tf}s bar", units="bps", rng=None,
             reason="Higher-timeframe context", **kw)
        emit(f"htf_range_bps_{tf}s", ops.bps(h - l_, c), desc=f"High-low range of the last COMPLETE {tf}s bar / close",
             units="bps", rng=(0, None), reason="Higher-timeframe volatility context", **kw)
        emit(f"htf_close_dist_bps_{tf}s", ops.bps(mid - c, mid),
             desc=f"Last mid minus the close of the last COMPLETE {tf}s bar", units="bps", rng=None,
             reason="Progress inside the current (incomplete) higher-timeframe bar, measured from known data", **kw)


def session(ctx: FeatureContext, cfg: FeatureConfig, prior: dict, emit: Emitter) -> None:
    g = f"{ctx.grid_s}s"
    s = session_arrays(ctx.start_ms, ctx.end_ms, ctx.mid_asof, ctx.col("mid_high"), ctx.col("mid_low"), ctx.grid_s,
                       ctx.rules)
    mid = ctx.mid_asof
    kw = dict(family="session", source="session", tf=g, lookback=0, leak=LEAK_STATE, nan="never NaN")
    emit("session_code", s["session_code"], desc="Primary session code: 0 ASIA, 1 LONDON, 2 NEW_YORK, 3 OVERLAP, "
         "4 OFF_HOURS, 5 MARKET_CLOSED (DST-aware local-time windows, evaluated at bar start)", units="code",
         rng=(0, 5), reason="Session context", **kw)
    for nm, desc in (("is_asia", "Asia session window"), ("is_london", "London session window"),
                     ("is_new_york", "New York session window"),
                     ("is_london_new_york_overlap", "London and New York windows both open")):
        emit(nm, s[nm], desc=f"1 inside the {desc} (0 when the nominal weekly market is closed)", units="flag",
             rng=(0, 1), reason="Session context", **kw)
    kw2 = dict(family="session", source="session", tf=g, lookback=0, leak=LEAK_SESSION,
               nan="NaN outside Asia/London/New York windows (no reference session)")
    emit("minutes_since_session_open", s["minutes_since_session_open"], desc="Minutes since the reference session opened "
         "(local wall-clock, DST-aware)", units="min", rng=(0, None), reason="Time-of-session effects", **kw2)
    emit("minutes_until_session_close", s["minutes_until_session_close"], desc="Minutes until the reference session "
         "closes", units="min", rng=(0, None), reason="Time-of-session effects", **kw2)
    hi, lo = s["session_high_so_far"], s["session_low_so_far"]
    emit("session_high_so_far", hi, desc="Highest mid since the reference session opened, up to the last complete bar "
         "only", units="price", rng=None, reason="Session extreme (causal)", **kw2)
    emit("session_low_so_far", lo, desc="Lowest mid since the reference session opened, up to the last complete bar "
         "only", units="price", rng=None, reason="Session extreme (causal)", **kw2)
    emit("session_range_so_far_bps", ops.bps(hi - lo, mid), desc="(session high so far - low so far) / mid",
         units="bps", rng=(0, None), reason="Session range development", **kw2)
    emit("dist_session_high_bps", ops.bps(hi - mid, mid), desc="Distance from the last mid to the session high so far",
         units="bps", rng=(0, None), reason="Position within the session range", **kw2)
    emit("dist_session_low_bps", ops.bps(mid - lo, mid), desc="Distance from the session low so far to the last mid",
         units="bps", rng=(0, None), reason="Position within the session range", **kw2)
    emit("dist_session_mean_bps", ops.bps(mid - s["session_mean_so_far"], mid),
         desc="(mid - mean of as-of mids since the session opened) / mid", units="bps", rng=None,
         reason="Distance from session mean (mean reversion)", **kw2)
    emit("session_so_far_complete", s["session_so_far_complete"], desc="1 if this session instance was observed from "
         "(about) its open; 0 if the data started mid-session so the 'so far' values are partial", units="flag",
         rng=(0, 1), reason="Marks partial session history explicitly", **kw2)


def regimes(ctx: FeatureContext, cfg: FeatureConfig, prior: dict, emit: Emitter) -> None:
    from fxscalp.regimes.spread_regime import classify_spread
    from fxscalp.regimes.volatility_regime import classify_volatility
    g = f"{ctx.grid_s}s"
    sc = classify_spread(prior, ctx.regime_cfg.spread)
    vc = classify_volatility(prior, ctx.regime_cfg.volatility)
    emit("spread_regime_code", sc, family="regime", desc="Spread regime: 0 UNKNOWN, 1 NORMAL, 2 ELEVATED, 3 EXTREME "
         "(relative/distribution-based; thresholds uncalibrated until real data)", source="regime_engine", tf=g,
         lookback=cfg.spread_hist_s, units="code", rng=(0, 3), nan="0 (UNKNOWN) when inputs are NaN",
         leak=LEAK_BACK, reason="Cost-regime gate input")
    emit("volatility_regime_code", vc, family="regime", desc="Volatility regime: 0 UNKNOWN, 1 LOW, 2 NORMAL, 3 HIGH, "
         "4 EXTREME (relative; uncalibrated until real data)", source="regime_engine", tf=g,
         lookback=cfg.vol_hist_s, units="code", rng=(0, 4), nan="0 (UNKNOWN) when inputs are NaN", leak=LEAK_BACK,
         reason="Volatility-regime context")


GROUPS: tuple[FeatureGroup, ...] = (
    FeatureGroup("micro", 1, micro), FeatureGroup("spread", 1, spread), FeatureGroup("structure", 1, structure),
    FeatureGroup("volatility", 1, volatility), FeatureGroup("momentum", 1, momentum),
    FeatureGroup("mean_reversion", 1, mean_reversion), FeatureGroup("mtf", 1, mtf), FeatureGroup("session", 1, session),
    FeatureGroup("regimes", 1, regimes),
)
