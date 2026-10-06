import datetime as dt

import numpy as np

from fxscalp.brokers.base import RAW_TICK_DTYPE
from fxscalp.features.normalize import from_raw_array
from fxscalp.features.pipeline import PipelineConfig, build_feature_frame

POINT = 0.01
UTC = dt.timezone.utc


def ms(*a):
    return int(dt.datetime(*a, tzinfo=UTC).timestamp() * 1000)


def raw(ts_ms, mid, spread=0.25):
    ts_ms = np.asarray(ts_ms, dtype="int64")
    mid = np.asarray(mid, dtype="float64")
    a = np.empty(len(ts_ms), dtype=RAW_TICK_DTYPE)
    a["time_msc"], a["time"] = ts_ms, ts_ms // 1000
    a["bid"], a["ask"] = np.round(mid - spread / 2, 4), np.round(mid + spread / 2, 4)
    a["last"], a["volume"], a["volume_real"], a["flags"] = 0.0, 0, 0.0, 6
    return a


def norm(arr, point=POINT, **kw):
    df, rep = from_raw_array(arr, point=point, **kw)
    return df


def features(arr, cfg=None, point=POINT):
    return build_feature_frame(norm(arr, point), cfg or PipelineConfig(), point=point)


def steady(start_utc_ms, seconds, per_sec=2, step=0.10, start_mid=2000.0, spread=0.25):
    """per_sec ticks at ms offsets 100 and 600 (2/s) in every second; mid rises by `step` each tick."""
    offs = np.array([100, 600] if per_sec == 2 else np.linspace(100, 900, per_sec).astype(int))
    ts = (start_utc_ms + (np.arange(seconds)[:, None] * 1000 + offs[None, :])).ravel()
    mid = start_mid + step * np.arange(len(ts))
    return raw(ts, mid, spread)
