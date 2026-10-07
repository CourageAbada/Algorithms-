"""Cost Model v0 (research). Observed quantities are used wherever they exist; everything else is an explicit parameter that
is UNKNOWN / UNCALIBRATED until verified, and labels that need it must be generated under a NAMED sensitivity scenario.

Execution convention (applies to every label): a LONG enters by BUYING at the ask and exits by SELLING at the bid; a SHORT
enters by SELLING at the bid and exits by BUYING at the ask. The observed bid/ask therefore carries the spread cost
(the entry spread is paid immediately; there is no separate exit spread term). Costs that bid/ask cannot show are:

  commission_round_turn_points   UNKNOWN / UNCALIBRATED   broker commission for the round turn, in points (1 point = 0.01 USD/oz)
  slippage_points_per_side       UNKNOWN / UNCALIBRATED   adverse fill vs the quoted price, in points, per side
  latency_s                      UNKNOWN / UNCALIBRATED   decision -> fill delay, quantised to the 1 s grid (0 or 1 s)

``extra_rt_points = commission_round_turn_points + 2 * slippage_points_per_side`` is subtracted from every net result and
widens every profit target. The numbers in SCENARIOS are HYPOTHETICAL sensitivity points (an order of magnitude for a raw-spread
account), NOT broker facts, and are never to be read as calibrated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

POINT_USD = 0.01          # XAUUSD point (verified from symbol metadata)


class UncalibratedCostError(ValueError):
    """A label was requested without an explicit scenario for a cost component whose value is UNKNOWN."""


@dataclass(frozen=True)
class CostScenario:
    name: str
    latency_s: int                     # 0 or 1 (1 s decision grid)
    commission_round_turn_points: float
    slippage_points_per_side: float
    status: str = "HYPOTHETICAL_SENSITIVITY_POINT_NOT_CALIBRATED"

    def __post_init__(self) -> None:
        if self.latency_s not in (0, 1):
            raise ValueError("latency is quantised to the 1 s grid: 0 or 1")
        if self.commission_round_turn_points < 0 or self.slippage_points_per_side < 0:
            raise ValueError("costs cannot be negative")

    @property
    def extra_rt_points(self) -> float:
        return self.commission_round_turn_points + 2.0 * self.slippage_points_per_side


SCENARIOS: dict[str, CostScenario] = {
    # explicit OPTIMISTIC lower bound: observed spread only. It is a sensitivity point, not an assumption that costs are zero.
    "C0_spread_only": CostScenario("C0_spread_only", 0, 0.0, 0.0),
    "C1_moderate": CostScenario("C1_moderate", 1, 7.0, 1.5),           # 7 + 2*1.5 = 10 extra points, 1 s latency
    "C2_pessimistic": CostScenario("C2_pessimistic", 1, 15.0, 7.5),    # 15 + 2*7.5 = 30 extra points, 1 s latency
}

COMPONENTS: dict[str, dict[str, Any]] = {
    "observed_spread": {"status": "OBSERVED", "source": "historical bid/ask of every tick (entry at ask/bid, exit at bid/ask)",
                        "value": None},
    "commission_round_turn_points": {"status": "UNKNOWN/UNCALIBRATED", "value": None, "note": "broker commission not verified"},
    "slippage_points_per_side": {"status": "UNKNOWN/UNCALIBRATED", "value": None, "note": "no fill data (DEMO feed, no orders)"},
    "latency_s": {"status": "UNKNOWN/UNCALIBRATED", "value": None, "note": "order latency not measured; terminal API latency is not execution latency"},
    "swap_financing": {"status": "NOT_MODELLED", "value": None, "note": "intraday horizons only"},
}


def resolve_scenario(scenario: CostScenario | str | None) -> CostScenario:
    """Labels must name a scenario: a silent default of 'no extra cost' is refused."""
    if scenario is None:
        raise UncalibratedCostError(
            "commission, slippage and latency are UNKNOWN/UNCALIBRATED: pass an explicit CostScenario (sensitivity point), "
            "never an implicit zero")
    if isinstance(scenario, str):
        if scenario not in SCENARIOS:
            raise UncalibratedCostError(f"unknown scenario {scenario!r}; known: {sorted(SCENARIOS)}")
        return SCENARIOS[scenario]
    return scenario


def cost_model_manifest() -> dict[str, Any]:
    return {"version": "cost_model/0", "point_usd": POINT_USD, "components": COMPONENTS,
            "execution_convention": "LONG: buy at ask, sell at bid; SHORT: sell at bid, buy at ask; spread cost is implicit in the bid/ask path",
            "extra_rt_points_formula": "commission_round_turn_points + 2 * slippage_points_per_side",
            "scenarios": {k: {**asdict(v), "extra_rt_points": v.extra_rt_points} for k, v in SCENARIOS.items()},
            "scenario_status": "HYPOTHETICAL sensitivity points, NOT broker facts, NOT calibrated",
            "rule": "unknown components must come from a named scenario; sensitivity analysis over all scenarios is mandatory for any reported result"}
