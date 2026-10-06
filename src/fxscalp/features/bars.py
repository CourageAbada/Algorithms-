"""Deterministic tick -> bar aggregation and multi-timeframe alignment.

Boundaries (explicit, UTC): a bar of ``tf_s`` seconds covers the half-open interval
``[k*tf_s, (k+1)*tf_s)`` in UTC epoch time. A tick exactly on a boundary belongs to the NEW bar. Bars are therefore
aligned to UTC multiples of the timeframe regardless of DST (DST only changes session labels, never bar edges).

Within a bar ticks are ordered by (timestamp, seq): open = first, close = last. Several ticks may share one timestamp.

Empty intervals are represented explicitly (``n_ticks == 0``, ``empty == True``, price columns NaN). Prices are never
forward-filled into bars. Gaps longer than the segment gap are not bars at all: the pipeline splits segments there.

Completeness: a bar is ``complete`` only when it can no longer receive ticks, i.e. data continues beyond its end. The final
bar of a dataset is NOT complete (the dataset may have been cut mid-bar) and is never used by alignment.

Alignment (anti look-ahead): ``align_completed`` maps each decision time ``t`` to the latest bar with ``end <= t``
that is complete. A bar still in progress at ``t`` is never visible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BAR_SCHEMA_VERSION = "bars/1"


def tick_derived(ts_ms: np.ndarray, bid: np.ndarray, ask: np.ndarray, mid: np.ndarray) -> pd.DataFrame:
    """Per-tick quantities that depend only on the tick and its predecessor (causal). The first tick of a segment has no
    predecessor: its returns/intervals are NaN and it is not counted as up/down/flat."""
    n = len(mid)
    d = pd.DataFrame(index=range(n))
    prev_mid = np.concatenate(([np.nan], mid[:-1]))
    with np.errstate(divide="ignore", invalid="ignore"):
        d["mid_return"] = np.log(mid / prev_mid)
        d["bid_return"] = np.log(bid / np.concatenate(([np.nan], bid[:-1])))
        d["ask_return"] = np.log(ask / np.concatenate(([np.nan], ask[:-1])))
    d["interval_ms"] = np.concatenate(([np.nan], np.diff(ts_ms).astype("float64")))
    dm = mid - prev_mid
    d["direction"] = np.where(np.isnan(dm), 0, np.sign(dm)).astype("int8")      # +1 up, -1 down, 0 flat/first
    d["has_prev"] = np.concatenate(([False], np.ones(max(n - 1, 0), bool)))
    # runs of consecutive same-direction NON-FLAT ticks; flat ticks neither extend nor break a run
    run = np.zeros(n, dtype="int64")
    nf = np.flatnonzero(d["direction"].to_numpy() != 0)
    same = np.zeros(n, bool)
    opp = np.zeros(n, bool)
    if len(nf):
        s = d["direction"].to_numpy()[nf].astype("int64")
        change = np.concatenate(([True], s[1:] != s[:-1]))
        start_pos = np.maximum.accumulate(np.where(change, np.arange(len(s)), 0))
        rl = (np.arange(len(s)) - start_pos + 1) * s                          # signed run length at each non-flat tick
        run[nf] = rl
        same[nf[1:]] = ~change[1:]
        opp[nf[1:]] = change[1:]
        # flat ticks inherit the run of the latest non-flat tick (as-of); ticks before the first non-flat have 0
        last_nf = np.maximum.accumulate(np.where(d["direction"].to_numpy() != 0, np.arange(n), -1))
        run = np.where(last_nf >= 0, run[np.maximum(last_nf, 0)], 0)
    d["run_len"] = run
    d["same_dir"] = same
    d["opp_dir"] = opp
    return d


def aggregate_bars(ts_ms: np.ndarray, bid: np.ndarray, ask: np.ndarray, volume: np.ndarray, tf_s: int, *,
                   data_continues_after: bool = False, micro: bool = False) -> pd.DataFrame:
    """Aggregate ONE segment of time-ordered ticks into bars of ``tf_s`` seconds (see module docstring)."""
    if tf_s <= 0:
        raise ValueError("timeframe must be positive")
    ts_ms = np.asarray(ts_ms, dtype="int64")
    n = len(ts_ms)
    if n == 0:
        return pd.DataFrame()
    if (np.diff(ts_ms) < 0).any():
        raise ValueError("aggregate_bars requires time-ordered ticks (normalisation sorts and counts reordering)")
    tf_ms = tf_s * 1000
    mid = (bid + ask) / 2.0
    spread = ask - bid
    td = tick_derived(ts_ms, bid, ask, mid)
    bi = ts_ms // tf_ms
    f = pd.DataFrame({"bi": bi, "mid": mid, "bid": bid, "ask": ask, "spread": spread, "vol": volume.astype("float64"),
                      "ts": ts_ms, "sq": np.nan_to_num(td["mid_return"].to_numpy() ** 2, nan=0.0)})
    if micro:
        hp = td["has_prev"].to_numpy()
        dirn = td["direction"].to_numpy()
        f["up"] = (hp & (dirn > 0)).astype("int64")
        f["down"] = (hp & (dirn < 0)).astype("int64")
        f["flat"] = (hp & (dirn == 0)).astype("int64")
        f["same"] = td["same_dir"].to_numpy().astype("int64")
        f["opp"] = td["opp_dir"].to_numpy().astype("int64")
        f["dt"] = np.nan_to_num(td["interval_ms"].to_numpy(), nan=0.0)
        f["n_dt"] = hp.astype("int64")
        f["run"] = td["run_len"].to_numpy().astype("float64")
        f["bid_ret"] = np.nan_to_num(td["bid_return"].to_numpy(), nan=0.0)
        f["ask_ret"] = np.nan_to_num(td["ask_return"].to_numpy(), nan=0.0)
    g = f.groupby("bi", sort=True)
    agg = pd.DataFrame({
        "n_ticks": g["mid"].size(),
        "mid_open": g["mid"].first(), "mid_high": g["mid"].max(), "mid_low": g["mid"].min(), "mid_close": g["mid"].last(),
        "bid_open": g["bid"].first(), "bid_high": g["bid"].max(), "bid_low": g["bid"].min(), "bid_close": g["bid"].last(),
        "ask_open": g["ask"].first(), "ask_high": g["ask"].max(), "ask_low": g["ask"].min(), "ask_close": g["ask"].last(),
        "spread_open": g["spread"].first(), "spread_high": g["spread"].max(), "spread_low": g["spread"].min(),
        "spread_close": g["spread"].last(), "spread_mean": g["spread"].mean(), "spread_median": g["spread"].median(),
        "spread_sum": g["spread"].sum(), "mid_sum": g["mid"].sum(), "volume_sum": g["vol"].sum(),
        "sum_sq_ret": g["sq"].sum(), "first_tick_ms": g["ts"].first(), "last_tick_ms": g["ts"].last()})
    if micro:
        for c, src in (("n_up", "up"), ("n_down", "down"), ("n_flat", "flat"), ("n_same", "same"), ("n_opp", "opp"),
                       ("sum_dt", "dt"), ("n_dt", "n_dt")):
            agg[c] = g[src].sum()
        agg["run_close"] = g["run"].last()
        agg["sum_bid_ret"] = g["bid_ret"].sum()
        agg["sum_ask_ret"] = g["ask_ret"].sum()
    full = np.arange(int(bi[0]), int(bi[-1]) + 1)
    agg = agg.reindex(full)                                                     # explicit empty intervals
    count_cols = [c for c in agg.columns if c in ("n_ticks", "n_up", "n_down", "n_flat", "n_same", "n_opp", "n_dt",
                                                  "sum_sq_ret", "sum_dt", "volume_sum", "spread_sum", "mid_sum",
                                                  "sum_bid_ret", "sum_ask_ret")]
    agg[count_cols] = agg[count_cols].fillna(0)
    agg["n_ticks"] = agg["n_ticks"].astype("int64")
    agg.insert(0, "start_ms", full * tf_ms)
    agg.insert(1, "end_ms", (full + 1) * tf_ms)
    agg.insert(2, "empty", agg["n_ticks"].to_numpy() == 0)
    last_ts = int(ts_ms[-1])
    agg.insert(3, "complete", (agg["end_ms"].to_numpy() <= last_ts) | bool(data_continues_after))
    agg["realized_vol"] = np.sqrt(agg["sum_sq_ret"])
    return agg.reset_index(drop=True)


def align_completed(decision_ms: np.ndarray, bars: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """For each decision time return (index of latest COMPLETE bar with end <= t, age_ms); index -1 when none.

    ``side='right'`` makes a bar whose end equals t visible (its ticks are all < t); a bar containing t (end > t) is not.
    """
    t = np.asarray(decision_ms, dtype="int64")
    if bars.empty:
        return np.full(len(t), -1), np.full(len(t), -1)
    ok = bars["complete"].to_numpy()
    ends = bars["end_ms"].to_numpy()
    pos = np.flatnonzero(ok)
    if pos.size == 0:                         # no complete bar yet at this timeframe
        return np.full(len(t), -1), np.full(len(t), -1)
    idx_in = np.searchsorted(ends[pos], t, side="right") - 1
    idx = np.where(idx_in >= 0, pos[np.maximum(idx_in, 0)], -1)
    age = np.where(idx >= 0, t - ends[np.maximum(idx, 0)], -1)
    return idx, age


def take(col: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """col[idx] with NaN where idx < 0."""
    col = np.asarray(col, dtype="float64")
    out = col[np.maximum(idx, 0)]
    return np.where(idx >= 0, out, np.nan)
