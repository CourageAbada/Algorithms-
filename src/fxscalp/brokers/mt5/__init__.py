"""MetaTrader 5 broker implementation (read-only in Phase 1). Only this package may import MetaTrader5."""

from fxscalp.brokers.mt5.adapter import AdapterState, HealthMonitor, MT5BrokerAdapter, Mt5Credentials

__all__ = ["AdapterState", "HealthMonitor", "MT5BrokerAdapter", "Mt5Credentials"]
