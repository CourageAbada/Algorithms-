# Forex / CFD Market Model

How this system models the market. Principle: **never assume a constant**; every instrument parameter comes from broker metadata discovered at runtime.

## 1. Pricing and venue

- OTC, decentralised: there is no single "market price". Each broker publishes its own bid/ask stream (aggregated from LPs, possibly marked up). Models trained on one feed are only valid for that feed (provenance recorded in dataset manifests).
- All fills and PnL use **bid/ask**: BUY enters at ask and exits at bid; SELL enters at bid and exits at ask. Mid is a feature/reference, never a fill price.
- Spread is variable and widens around news, rollover, and thin liquidity.
- `volume` in MT5 FX/CFD ticks is typically **tick volume** (update count), not traded volume. Treated as an activity proxy only.

## 2. Instrument metadata (from `SymbolInfo`)

`digits, point, tick_size, tick_value (profit & loss variants), contract_size, volume_min/max/step, stops_level, freeze_level, swap_long/short (+ swap mode, 3-day swap day), margin currency/mode, execution mode, filling modes, trade mode, session times`. Cached with a timestamp and refreshed on connect and daily.

## 3. Money maths

- `pnl = (exit_price - entry_price) * direction * volume * contract_size` expressed in profit currency, converted to account currency via broker-provided tick value (or an explicit conversion rate); **do not hardcode pip value**.
- Pip is a convention, not a unit we compute with. Internal unit: **points / price distance in instrument price units**, converted with `tick_size`/`tick_value`.
- Commission: per-lot per-side (account-specific), configured or read from deal history.
- Swap: applied when a position is held across the rollover; modelled in backtest with the symbol's swap parameters and the triple-swap weekday.
- Margin: `order_calc_margin` via adapter; own formula used as cross-check in tests.

## 4. Time

- Internal: UTC (tz-aware). Broker server time offset derived per broker (including DST rules of the server) and verified against a known reference tick; unit-tested across DST transitions. See `SessionEngine`.
- Sessions (UTC reference windows, **configurable and to be validated from data**, not constants): ASIA, LONDON, NEW_YORK, LONDON_NEW_YORK_OVERLAP, ROLLOVER (around 21:00-23:00 UTC, shifts with US DST), OFF_HOURS. Weekend gap: Friday close to Sunday open; no features may span the gap without a gap flag.

## 5. Execution realities

Stop level / freeze level restrict SL/TP distance and modification; filling mode (FOK/IOC/RETURN) varies by symbol and broker; execution mode (market/instant/request) changes requote behaviour; slippage and rejections occur; partial fills possible; netting vs hedging account changes position semantics. See `EXECUTION_MODEL.md`.

## 6. Instrument classes

| | XAUUSD | EURUSD | GBPUSD |
|---|---|---|---|
| Class | Metal CFD/spot | FX major | FX major |
| Contract | broker-defined (often 100 oz/lot) | usually 100,000 | usually 100,000 |
| Typical spread | wide in points, varies strongly | tight | tight, wider than EUR in thin hours |
| Volatility | high, regime-dependent | moderate | moderate-high |
| Notes | USD/real-yield/risk sensitivity; spike risk around US data | London/NY driven | GBP news (BoE, UK CPI) |

"Typical" values are qualitative and must be measured per broker (Phase 2).
