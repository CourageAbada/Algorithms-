"""DELIBERATELY LEAKY feature implementations. They exist only to prove that the leakage tests detect each class of
mistake. Never import these from production code. Signature matches the pipeline `extra` hook: (ctx, prior) -> columns."""
import numpy as np
import pandas as pd

from fxscalp.features.bars import take


def bad_future_bar(ctx, prior):
    """Aligns by bar START, so the bar still in progress (with its FINAL close) is visible: classic MTF look-ahead."""
    bars = ctx.mtf[300]
    idx = np.searchsorted(bars["start_ms"].to_numpy(), ctx.end_ms - 1, side="right") - 1
    return {"bad_future_bar": take(bars["mid_close"].to_numpy(), idx)}


def bad_future_session_extreme(ctx, prior):
    """Final high of the data/session instead of the running high."""
    return {"bad_session_high": np.full(ctx.n, np.nanmax(ctx.col("mid_high")))}


def bad_centered_window(ctx, prior):
    return {"bad_centered": pd.Series(ctx.mid_asof).rolling(61, center=True, min_periods=1).mean().to_numpy()}


def bad_negative_shift(ctx, prior):
    return {"bad_neg_shift": pd.Series(ctx.mid_asof).shift(-5).to_numpy()}


def bad_forward_asof(ctx, prior):
    """merge_asof in the wrong direction: attaches the NEXT bar's close."""
    bars = ctx.mtf[60]
    left = pd.DataFrame({"t": ctx.end_ms})
    right = pd.DataFrame({"t": bars["end_ms"].to_numpy(), "c": bars["mid_close"].to_numpy()})
    return {"bad_forward_asof": pd.merge_asof(left, right, on="t", direction="forward")["c"].to_numpy()}


def bad_backward_asof_on_start(ctx, prior):
    """Backward as-of but keyed on the bar START time: still exposes the in-progress bar's final close."""
    bars = ctx.mtf[60]
    left = pd.DataFrame({"t": ctx.end_ms - 1})
    right = pd.DataFrame({"t": bars["start_ms"].to_numpy(), "c": bars["mid_close"].to_numpy()})
    return {"bad_asof_start": pd.merge_asof(left, right, on="t", direction="backward")["c"].to_numpy()}


def bad_future_normalization(ctx, prior):
    x = prior["ret_bps_60s"]
    return {"bad_future_norm": (x - np.nanmean(x)) / np.nanstd(x)}


class WholeDatasetScaler:
    """Ignores the training boundary: fits on EVERYTHING it is given."""

    def __init__(self, fit_end_ms):
        self.fit_end_ms, self.params = fit_end_ms, {}

    def fit(self, frame, cols):
        for c in cols:
            x = frame[c].to_numpy(dtype="float64")
            x = x[np.isfinite(x)]
            self.params[c] = (float(np.median(x)), float(np.quantile(x, 0.75) - np.quantile(x, 0.25)) or 1.0)
        return self


def bad_whole_dataset_scaler(ctx, prior):
    x = pd.DataFrame({"x": prior["ret_bps_60s"], "timestamp_utc_ms": ctx.end_ms})
    s = WholeDatasetScaler(0).fit(x, ["x"])
    med, sc = s.params["x"]
    return {"bad_scaled": (x["x"].to_numpy() - med) / sc}


def bad_future_volatility(ctx, prior):
    r = pd.Series(np.log(ctx.mid_asof)).diff()
    return {"bad_future_vol": r.rolling(60).std().shift(-60).to_numpy()}


def bad_reversed_rolling(ctx, prior):
    r = pd.Series(ctx.mid_asof)[::-1]
    return {"bad_reversed": r.rolling(30, min_periods=1).max()[::-1].to_numpy()}


def bad_target_contamination(ctx, prior):
    """A 'feature' that is the label: the next-30s return."""
    m = pd.Series(np.log(ctx.mid_asof))
    return {"bad_target": (m.shift(-30) - m).to_numpy() * 1e4}


ALL_BAD = {
    "future_bar": bad_future_bar, "future_session_extreme": bad_future_session_extreme, "centered_window": bad_centered_window,
    "negative_shift": bad_negative_shift, "forward_asof": bad_forward_asof, "backward_asof_on_start": bad_backward_asof_on_start,
    "future_normalization": bad_future_normalization, "whole_dataset_scaler": bad_whole_dataset_scaler,
    "future_volatility": bad_future_volatility, "reversed_rolling": bad_reversed_rolling,
    "target_contamination": bad_target_contamination,
}
