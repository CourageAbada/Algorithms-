"""Deterministic canonical-instrument -> broker-symbol discovery and the persisted mapping.

Broker-specific names (XAUUSDm, XAUUSD.a, mXAUUSD, GOLD...) live ONLY in the generated mapping file
(configs/broker_symbols/<broker>__<server>.yaml); application code always uses canonical names
(XAU_USD, EUR_USD, GBP_USD) and resolves them through ``SymbolMap``.

Algorithm (pure, deterministic, no I/O):
  tier 0  exact alias match (case-sensitive)
  tier 1  exact, case-insensitive
  tier 2  separator-insensitive match ("XAU/USD", "xau_usd")
  tier 3  alias plus a RECOGNISED broker prefix/suffix hint (m, c, z, pro, ecn, raw, i, r, a, micro, #, ., _)
  tier 4  alias plus any short alphanumeric affix -> NEVER auto-selected; reported for human review
The best tier wins; if two or more symbols tie it is AMBIGUOUS and an explicit pin is required. Symbols
whose reported base/profit currency contradict the instrument are rejected. Nothing is guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from fxscalp.brokers.base import SymbolInfo
from fxscalp.brokers.errors import AmbiguousSymbolError, SymbolNotFoundError
from fxscalp.core.provenance import slug

MAP_SCHEMA_VERSION = "symbol_map/1"
_SEP = re.compile(r"[^A-Za-z0-9]")
_HINTS = {"m", "c", "z", "pro", "ecn", "raw", "i", "r", "a", "micro", "mini", "std", "x", "b", "s", "f", "k"}
_CCY = {"USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD", "XAU", "XAG", "CNH", "SEK", "NOK", "MXN", "ZAR",
        "TRY", "SGD", "HKD", "PLN", "CZK", "HUF", "BTC", "ETH"}


@dataclass(frozen=True)
class InstrumentSpec:
    canonical: str
    aliases: tuple[str, ...]
    expected_base: str | None = None
    expected_profit: str | None = None

    @staticmethod
    def from_config(cfg: Mapping[str, Any]) -> "InstrumentSpec":
        return InstrumentSpec(cfg["canonical"], tuple(cfg["aliases"]), cfg.get("expected_base_currency"),
                              cfg.get("expected_profit_currency"))


@dataclass(frozen=True)
class Candidate:
    name: str
    tier: int
    reason: str
    rejected: str | None = None       # reason when excluded by metadata validation
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class DiscoveryResult:
    canonical: str
    status: str                        # "unique" | "pinned" | "ambiguous" | "needs_review" | "not_found"
    chosen: str | None
    candidates: tuple[Candidate, ...]
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status in ("unique", "pinned") and self.chosen is not None


def _core(s: str) -> str:
    return _SEP.sub("", s).upper()


def _affix_tier(symbol: str, alias: str) -> tuple[int, str] | None:
    """Tier of ``symbol`` against one alias, or None if unrelated."""
    if symbol == alias:
        return 0, "exact alias"
    if symbol.lower() == alias.lower():
        return 1, "case-insensitive alias"
    sc, ac = _core(symbol), _core(alias)
    if sc == ac:
        return 2, "separator-insensitive alias"
    # symbol = prefix + alias-core + suffix, built from alphanumeric runs split on separators
    idx = sc.find(ac)
    if idx < 0:
        return None
    # recover the affix text using the original characters
    raw_upper = symbol.upper()
    pos = raw_upper.find(ac)
    if pos < 0:
        # alias core spans separators in the original (e.g. "XAU.USDm"): fall back to normalised affixes
        pre, suf = sc[:idx], sc[idx + len(ac):]
    else:
        pre, suf = raw_upper[:pos], raw_upper[pos + len(ac):]
    pre_t, suf_t = _SEP.sub("", pre).lower(), _SEP.sub("", suf).lower()
    if len(pre_t) > 6 or len(suf_t) > 6:
        return None
    # a different currency code glued on means a different instrument (GOLDEUR != GOLD)
    for part in (pre_t.upper(), suf_t.upper()):
        if part and any(part.startswith(c) or part.endswith(c) for c in _CCY if len(part) >= 3):
            return None
    if (not pre_t or pre_t in _HINTS) and (not suf_t or suf_t in _HINTS):
        return 3, f"alias with broker affix prefix={pre!r} suffix={suf!r}"
    return 4, f"alias with unrecognised affix prefix={pre!r} suffix={suf!r}"


def discover(spec: InstrumentSpec, symbols: Iterable[SymbolInfo | str], *, pin: str | None = None) -> DiscoveryResult:
    infos: dict[str, SymbolInfo | None] = {}
    for s in symbols:
        if isinstance(s, str):
            infos[s] = None
        else:
            infos[s.broker_symbol] = s
    cands: list[Candidate] = []
    for name, info in sorted(infos.items()):
        best: tuple[int, str] | None = None
        for alias in spec.aliases:
            t = _affix_tier(name, alias)
            if t and (best is None or t[0] < best[0]):
                best = t
        if best is None:
            continue
        rejected, warns = None, []
        if info is not None:
            for want, got, label in ((spec.expected_base, info.currency_base, "base"),
                                     (spec.expected_profit, info.currency_profit, "profit")):
                if want and got:
                    if got.upper() != want.upper():
                        rejected = f"{label} currency {got!r} != expected {want!r}"
                elif want and not got:
                    warns.append(f"{label} currency not reported by broker")
            if info.trade_mode == 0:
                warns.append("trade_mode=DISABLED")
        cands.append(Candidate(name, best[0], best[1], rejected, tuple(warns)))
    if pin is not None:
        match = [c for c in cands if c.name == pin]
        if not match:
            return DiscoveryResult(spec.canonical, "not_found", None, tuple(cands),
                                   f"pinned symbol {pin!r} is not offered by this server or does not match {spec.canonical}")
        if match[0].rejected:
            return DiscoveryResult(spec.canonical, "not_found", None, tuple(cands),
                                   f"pinned symbol {pin!r} rejected: {match[0].rejected}")
        return DiscoveryResult(spec.canonical, "pinned", pin, tuple(cands), "explicit pin in configuration")
    valid = [c for c in cands if not c.rejected]
    if not valid:
        return DiscoveryResult(spec.canonical, "not_found", None, tuple(cands),
                               f"no broker symbol matches {spec.canonical} (aliases {list(spec.aliases)})")
    best_tier = min(c.tier for c in valid)
    top = [c for c in valid if c.tier == best_tier]
    if best_tier >= 4:
        return DiscoveryResult(spec.canonical, "needs_review", None, tuple(cands),
                               "only unrecognised-affix matches found; pin one explicitly after review")
    if len(top) > 1:
        return DiscoveryResult(spec.canonical, "ambiguous", None, tuple(cands),
                               f"{len(top)} symbols tie at tier {best_tier}: {[c.name for c in top]}; pin one in "
                               "the symbol map")
    return DiscoveryResult(spec.canonical, "unique", top[0].name, tuple(cands), top[0].reason)


# --------------------------------------------------------------------------------------------
# Persisted mapping
# --------------------------------------------------------------------------------------------
def map_path(config_dir: Path, broker: str, server: str) -> Path:
    return config_dir / "broker_symbols" / f"{slug(broker)}__{slug(server)}.yaml"


def build_map_document(broker: str, server: str, results: Sequence[DiscoveryResult],
                       pins: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Deterministic document (no timestamps) so identical discovery yields an identical file."""
    mappings: dict[str, Any] = {}
    for r in sorted(results, key=lambda x: x.canonical):
        entry: dict[str, Any] = {"status": r.status, "broker_symbol": r.chosen, "detail": r.detail,
                                 "candidates": [{"name": c.name, "tier": c.tier,
                                                 **({"rejected": c.rejected} if c.rejected else {})}
                                                for c in r.candidates]}
        mappings[r.canonical] = entry
    doc: dict[str, Any] = {"schema_version": MAP_SCHEMA_VERSION, "broker": broker, "server": server,
                           "mappings": mappings}
    if pins:
        doc["pins"] = dict(sorted(pins.items()))
    return doc


def write_map(path: Path, doc: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(doc), sort_keys=True, default_flow_style=False), encoding="utf-8")


class SymbolMap:
    """canonical -> broker symbol, loaded from the persisted mapping. The only place names are resolved."""

    def __init__(self, doc: Mapping[str, Any]):
        if doc.get("schema_version") != MAP_SCHEMA_VERSION:
            raise ValueError(f"unsupported symbol map schema {doc.get('schema_version')!r}")
        self.doc = dict(doc)

    @staticmethod
    def load(path: Path) -> "SymbolMap":
        return SymbolMap(yaml.safe_load(path.read_text(encoding="utf-8")))

    @property
    def pins(self) -> dict[str, str]:
        return dict(self.doc.get("pins", {}))

    def broker_symbol(self, canonical: str) -> str:
        m = self.doc["mappings"].get(canonical)
        if m is None or not m.get("broker_symbol"):
            status = None if m is None else m.get("status")
            if status == "ambiguous":
                raise AmbiguousSymbolError(f"{canonical}: ambiguous broker symbols; add a pin to {self.doc.get('broker')} map")
            raise SymbolNotFoundError(f"{canonical}: no broker symbol mapped on {self.doc.get('server')!r} "
                                      f"(status={status}); run discovery (scripts.verify_mt5) and review the map")
        return str(m["broker_symbol"])
