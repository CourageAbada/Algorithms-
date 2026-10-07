"""Deterministic selection of representative days from the per-day profile (descriptive extremes only)."""

from __future__ import annotations

import pandas as pd


def pick_days(df: pd.DataFrame) -> dict[str, str]:
    td = df[df["n_ticks"] > 0].copy()
    med = td["n_ticks"].median()
    used: set[str] = set()
    out: dict[str, str] = {}

    def choose(label: str, frame: pd.DataFrame, col: str, how: str) -> None:
        f = frame[~frame["day"].isin(used)].dropna(subset=[col])
        if f.empty:
            return
        if how == "max":
            row = f.loc[f[col].idxmax()]
        elif how == "min":
            row = f.loc[f[col].idxmin()]
        else:
            row = f.loc[(f[col] - med).abs().idxmin()]
        out[label] = row["day"]
        used.add(row["day"])

    mid_week = td[td["weekday"].isin(["Tue", "Wed", "Thu"])]
    choose("normal_activity", mid_week, "n_ticks", "median")
    choose("high_volatility", td, "range_bps", "max")
    choose("wide_spread", td, "spread_p99", "max")
    choose("widest_single_event_rollover_window", td, "spread_max", "max")
    choose("low_activity", td[td["n_ticks"] >= 20_000], "n_ticks", "min")
    if "week_open" in td:
        wo = td[td["week_open"].fillna(False).astype(bool)]
        choose("week_open", wo, "n_ticks", "median")
    if "week_close" in td:
        wc = td[td["week_close"].fillna(False).astype(bool)]
        choose("week_close", wc, "n_ticks", "median")
    return out
