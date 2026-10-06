# Risk Model

The risk engine is a separate package. It consumes a `TradeProposal` (plain numbers) plus market/account/system state and returns a `RiskDecision`. It does not import ML code, it cannot be bypassed by the models, the signal layer or the LLM, and its limits are human-owned configuration (`configs/risk*.yaml`) that no automated process writes.

## 1. Pre-trade checks (all must pass, evaluated in order; each failure is a reason code)

| Check | Notes |
|---|---|
| HALT_NEW_TRADES not engaged (and EMERGENCY_FLATTEN not engaged) | Sentinel file (existence alone halts, malformed content still halts), API/in-process engage; release only via CLI with confirmation |
| Mode allows the action | e.g. no order sink in SHADOW |
| Data health | tick age <= `max_data_staleness_ms`; no gap; clock sane; feed rate > floor |
| Broker health | connected, account verified (demo/live as configured), symbol tradable, market open |
| Model health | validated champion loaded, not expired, novelty gate OK, prediction finite |
| Session/news | session allowed for instrument; event policy NORMAL / REDUCED_RISK / NO_NEW_TRADES |
| Spread | spread within dynamic limit (percentile / ATR-fraction / absolute cap) |
| Expected edge | net edge > min_edge |
| Confidence | calibrated meta-probability >= validated threshold |
| Account limits | daily loss, max drawdown (from peak equity), consecutive losses, cooldowns, max positions, max exposure, max leverage |
| Latency | recent decision->ack latency <= limit |
| Slippage | recent realised slippage within limit |
| Broker rules | stop level, freeze level, lot step/min/max, margin sufficient (`order_calc_margin` + buffer) |

Any failure -> `REJECT`. Default on error/unknown -> `REJECT`.

## 2. Position sizing

```
risk_amount      = equity * risk_per_trade_pct            (account currency)
stop_distance    = |entry - SL|                            (price units; entry = ask for BUY, bid for SELL)
loss_per_lot     = (stop_distance / tick_size) * tick_value_loss
                   + commission_per_lot (round trip) + expected_slippage_cost_per_lot
raw_lots         = risk_amount / loss_per_lot
lots             = floor_to_step(raw_lots, volume_step)
lots             = min(lots, volume_max, max_lots_config, margin_limited_lots)
if lots < volume_min:  REJECT ("min lot exceeds risk budget")   # never round up into more risk
```
Notes: `tick_value_loss` is the loss-side tick value; if only `tick_value` exists use it with conversion tested. Spread is already realised at entry price, so SL distance is measured from the actual fill reference. Rounding is always **down**. Currency conversion to account currency uses broker-provided values, tested with account currency != profit currency. 0.01 lot is never assumed to have fixed risk.

## 3. Stops, targets, exits (all candidates tested; none assumed)

ATR/vol-adjusted SL/TP, structure stops, time stops, trailing, break-even, partials. Break-even and trailing are evaluated on identical entries (E8) and adopted only if net expectancy improves out of sample. Broker-side SL/TP are always attached to new orders so a client crash does not leave a naked position. Exit logic modifications respect stop/freeze levels.

## 4. Account-level controls

Max daily loss (realised + unrealised, by trading day in a configured timezone), max drawdown from equity peak, max consecutive losses, cooldown after loss, cooldown after abnormal volatility, max simultaneous positions, max exposure/leverage, max trades per hour/day. Breach -> NO NEW TRADES and cooldown/halting until a human reset where specified.

## 5. Safety states (decision D-7): two independent concepts

**HALT_NEW_TRADES** (global kill switch for new exposure). Engage sources: manual (dashboard/CLI/sentinel file), automatic (daily loss, drawdown, consecutive losses, data/broker/time-base failure beyond grace, repeated order rejects, model/drift/novelty failure, reconciliation mismatch, DB/journal outage). Effect: block new entries immediately; open positions are left to their broker-side SL/TP. Cheap to engage; any component may do so.

**EMERGENCY_FLATTEN** (controlled closure of existing exposure). Deliberately hard to trigger: only (a) an explicit authorised command carrying a token from a human-controlled secret (no configured token = cannot be authorised), or (b) a human-configured hard-loss trigger (e.g. an equity-drawdown threshold well beyond the HALT thresholds; value set by the human, not by this project). Never triggered by ordinary health checks, drift alarms or model failures. Effect: halts new trades, then closes positions **by position ticket** with bounded retries, logs each close, reconciles afterwards, and reports any position that could not be closed. It must not send blind opposite-side orders (hedge risk on hedging accounts). It can only reduce exposure.

Both are implemented as pure state logic in `src/fxscalp/core/safety.py` (tested); closing is done by the execution layer. Every transition goes to the audit ledger. The earlier proposal D-7 "block-only vs flatten" is replaced by this two-concept design.

## 6. Risk events

Every veto/reduction/limit breach is journaled with reason code, inputs and timestamp, enabling analysis of what the risk layer is blocking.

## 7. Defaults

Config placeholders in `configs/risk.example.yaml` are conservative placeholders, **not** recommendations. No leverage or risk percentage is validated until demo evidence exists.

## 8. Tests required (Phase 4 onward, before any order path)

Property tests for sizing (lots * loss_per_lot <= risk_amount for random metadata; rounding down; min-lot rejection); table tests per symbol metadata (XAUUSD 2/3 digits, EURUSD 5 digits, JPY-quote, account currencies); every veto path; kill-switch persistence across restarts; fail-closed on missing inputs.
