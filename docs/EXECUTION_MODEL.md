# Execution Model

## 1. Components

`execution/` owns the order lifecycle state machine; `brokers/` provides `BrokerAdapter`; `execution/sinks` has `SimulatedSink`, `ShadowSink`, `BrokerSink`. Only `BrokerSink` can reach `submit_order`, and only if `mode.may_send_orders`, the account check passes, and the `RiskDecision` is `APPROVE`.

## 2. Order lifecycle

`PROPOSED -> RISK_APPROVED -> PRECHECKED (order_check) -> SUBMITTED -> ACKNOWLEDGED -> (PARTIALLY_)FILLED -> OPEN -> CLOSING -> CLOSED` and terminal `REJECTED / CANCELLED / EXPIRED / ERROR`. Each transition is timestamped and persisted. Orders carry a client tag (idempotency key) so after a restart the engine reconciles broker positions/orders with its own state before doing anything else (cf. Vibe-Trading's client-order-id recovery concept). Reconciliation mismatch -> kill switch.

## 3. Pre-submit validation

Symbol tradable; market open; price fresh (quote age limit); SL/TP respect stops level; modifications respect freeze level; volume on lot step and within min/max; margin sufficient; deviation (max slippage in points) set; filling mode valid for the symbol (selected from `symbol_info`); `order_check` result OK. Failure -> no order, journaled.

## 4. Broker-specific behaviours to handle

Execution modes (market/instant/request/exchange); filling modes FOK/IOC/RETURN; requotes; off-quotes; invalid stops; `TRADE_RETCODE_*` mapping to a neutral error enum; partial fills; netting vs hedging accounts (position IDs); trade-context-busy retries (bounded, never blind-resubmit without checking state to avoid duplicate orders); terminal disconnect during submit (unknown outcome -> reconcile from positions/history before any retry).

## 5. Execution-quality record (per attempted trade)

decision timestamp, requested price, bid, ask, spread at decision, requested lot, requested SL/TP, submission timestamp, broker ack timestamp, fill timestamp, fill price, slippage (points and in ATR/cost units), commission, retcode, result/PnL. Derived metrics: signal-to-order latency, ack latency, fill latency, slippage distribution (by session/spread/volatility), rejection rate. Feeds back into (a) the backtest slippage/latency model, (b) risk limits (latency/slippage gates), (c) drift monitors.

## 6. Spread filter

Dynamic: compare current spread with (i) rolling distribution for the same instrument/session (percentile), (ii) volatility-adjusted ratio `spread/ATR`, (iii) absolute instrument cap. Block on abnormal values, rapid widening, or stale quotes. Thresholds chosen from data (E1/E2), per instrument.

## 7. Latency budget (to be measured, Phase 1)

tick arrival -> feature -> prediction -> risk -> order submit -> ack. Targets are set after measurement; trades are blocked when recent latency exceeds the limit because the edge assumption includes execution delay.

## 8. Exit management

Broker-side SL/TP always; optional engine-managed trailing/break-even/time-stop via `modify_order` / `close_position` (tested features, not defaults). `close_position` always targets a **position ticket** (`TRADE_ACTION_DEAL` with `position=<ticket>` or `CLOSE_BY`), never a blind opposite order: on a hedging account an opposite order opens a hedge (secondary evidence, to be confirmed on a demo hedging account in Phase 1). HALT_NEW_TRADES leaves positions to their SL/TP; EMERGENCY_FLATTEN (authorised only, see `RISK_MODEL.md` §5) closes by ticket with bounded retries, per-position logging and a final reconciliation.

## 8a. MT5 session and call discipline (Phase 0.5)

One long-lived terminal session (attach once), all calls on one owner thread behind a lock; no per-call `initialize()/shutdown()`; never use the package's bundled `Buy/Sell/Close` helpers (fixed deviation 10, blind retries). Every order: `order_check` first, then `order_send`; filling mode chosen from a per-symbol probe recorded in the verification report (the `SYMBOL_FILLING_*` bit values are unverified); all `TRADE_RETCODE_*` mapped to a neutral enum; unknown outcome after a disconnect is reconciled from positions/deal history before any retry.

## 9. Shadow and demo parity

ShadowSink replicates fill logic using live bid/ask (+ modelled latency/slippage) for hypothetical fills and PnL; it never holds a broker adapter. Demo/Shadow/Replay/Backtest comparisons on the same period are used to measure modelling error in the simulator.
