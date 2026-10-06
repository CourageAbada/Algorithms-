# Feature Catalog

Rules: (1) every feature has a **hypothesis** and an owner-defined **lookback**; (2) computed from information available **at or before decision time t** (leakage test enforces: value at t from `data[:t]` equals value at t from full data); (3) normalised to be comparable across sessions/volatility (ATR, percentile, z-score on rolling past window); (4) versioned; (5) kept only if ablation shows incremental out-of-sample value (E4). "Status" = Proposed until tested. No feature below is claimed to be predictive.

Notation: `m` mid, `s` spread, `W` window (ticks or seconds), `ATR_n` ATR on resolution n.

## A. Spread / cost

| Feature | Definition | Hypothesis |
|---|---|---|
| spread_abs, spread_rel | `ask-bid`, `s/m` | Cost proxy; high spread = poor expectancy |
| spread_pctile_sess | Rank of s in trailing distribution for the same session | Abnormal vs normal for the time of day |
| spread_expansion | `s / median_W(s)` | Liquidity withdrawal / event onset |
| spread_to_atr | `s / ATR_1m` | Volatility-adjusted cost |
| spread_contraction_rate | slope of s over W | Liquidity returning |

## B. Tick microstructure

| Feature | Definition | Hypothesis |
|---|---|---|
| bid_vel, ask_vel, mid_vel | `d(price)/dt` over W (EWMA) | Short-term momentum |
| tick_rate | ticks/s over W | Activity/information arrival |
| tick_accel | change in tick_rate | Burst onset |
| up_down_ratio | `#upticks/(#up+#down)` of mid over last N ticks | Directional pressure |
| signed_tick_imbalance | `sum(sign(dm))/N` | As above, signed |
| rv_short | sqrt(sum of squared mid log-returns) over W | Short-term volatility |
| price_accel | second difference of EWMA mid | Momentum change |
| burst_flag | `|ret_W| / rv_baseline > k` | Momentum burst vs noise |
| quote_stale_ms | time since last quote change | Feed health / illiquidity |

## C. Bar-based (multi-resolution)

Returns (log) over {5s,15s,1m,5m,15m,1h}; EMA/SMA distances normalised by ATR; RSI; MACD histogram (normalised); ATR and ATR ratio (short/long); ADX; stochastic; Bollinger %B and bandwidth; realised vol ratios; VWAP-like distance (tick-volume weighted, labelled as such). Hypothesis tested per family, not per indicator; **indicator soup is prohibited**: each family must pass ablation.

## D. Mean reversion / structure

| Feature | Definition |
|---|---|
| dist_mean_W | `(m - mean_W)/ATR` (z-score vs rolling mean) |
| dist_local_high/low | `(HH_W - m)/ATR`, `(m - LL_W)/ATR` |
| range_compression | `range_W_short / range_W_long` |
| range_expansion | inverse / breakout of last range |
| breakout_pressure | distance beyond range boundary / ATR, times persistence count |
| swing_hi/lo | fractal pivot: bar `i` is a swing high if `high_i > high_{i±k}` for k=1..K; **confirmed only after K bars** (lag explicit) |
| HH/HL/LH/LL state | sequence of last two confirmed swings |
| failed_breakout | price exceeded confirmed swing/range boundary then closed back inside within M bars |
| trend_strength | regression slope t-stat / ADX, over W |
| vol_compression/expansion | ATR percentile bands, Bollinger bandwidth percentile |

Subjective chart terms (e.g. "order block", "liquidity sweep") are not implemented unless given a precise mathematical definition and tested.

## E. Session / time

session id, minutes since open, minutes to close, overlap flag, session high/low so far, distance to them (ATR-normalised), session range so far vs trailing session range, hour-of-day and weekday (cyclic encoding), minutes to/since rollover, weekend-gap flag.

## F. Event risk

Minutes to next / since last high-impact event (by currency, USD-weighted for XAU), event impact class, surprise magnitude (only after release, using actual vs forecast, point-in-time).

## G. Depth (experimental)

bid/ask depth, imbalance, concentration, depth change. Only if the feed is verified reliable; labelled `broker_dom` and never assumed to represent the full market.

## H. Context for meta-model

Primary-model confidence, regime, recent realised performance (trailing trades: rolling expectancy/loss streak, computed strictly from closed trades), spread state, execution latency/slippage recent history, news proximity, trend alignment across timeframes.

## I. Cross-instrument (later)

EURUSD/GBPUSD returns and DXY-like synthetic index for XAUUSD **only** where timestamps align on the same feed.

## Implementation notes

Incremental O(1) updates with ring buffers; same code path for batch (vectorised, tested equal to incremental) and live; NaN policy: warm-up rows are explicitly invalid and produce NO_TRADE; every output carries `feature_version`.
