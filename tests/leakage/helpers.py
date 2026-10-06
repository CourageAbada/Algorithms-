import numpy as np
import pandas as pd

from fxscalp.brokers.base import RAW_TICK_DTYPE
from fxscalp.features.normalize import from_raw_array
from fxscalp.features.pipeline import PipelineConfig, build_feature_frame
from fxscalp.synthetic.xauusd import SynthConfig, generate_ticks

POINT = 0.01


def raw_to_df(arr: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({n: arr[n] for n in arr.dtype.names})
    df.insert(0, "seq", np.arange(len(arr)))
    return df


def df_to_raw(df: pd.DataFrame) -> np.ndarray:
    a = np.empty(len(df), dtype=RAW_TICK_DTYPE)
    for n in RAW_TICK_DTYPE.names:
        a[n] = df[n].to_numpy()
    return a


def make_compute(extra=None, cfg=None):
    """raw tick frame (arrival order) -> normalise (+quality quarantine) -> features. The WHOLE chain is under test."""
    def compute(df: pd.DataFrame) -> pd.DataFrame:
        ticks, _ = from_raw_array(df_to_raw(df), point=POINT)
        return build_feature_frame(ticks, cfg or PipelineConfig(), point=POINT, extra=extra).df
    return compute


def cutoffs(n: int, fracs=(0.15, 0.37, 0.52, 0.8, 0.97)) -> list[int]:
    return [int(n * f) + 7 for f in fracs]


ANOMALY_CFG = dict(dup_prob=0.002, out_of_order_prob=0.001, outages=((5400, 100),), spread_widening=((6000, 400, 5.0),))


def scenario(**kw) -> pd.DataFrame:
    return raw_to_df(generate_ticks(SynthConfig(**kw)))
