# Third-Party Notices and Licensing Record

This repository is **private/proprietary** (decision D-9). Obligations from third-party material are recorded here.

## Code copied into this repository

**None** (confirmed at the end of Phase 1). The Vibe-Trading vendoring proposed in Phase 0.5 was deferred by the reviewer to the validation subsystem; no FreqAI or Qlib code was copied. Planned vendored components (each will carry a header with origin URL, commit and licence, and an entry below when added):

| Planned component | Origin | Licence | Pinned commit | Status |
|---|---|---|---|---|
| `crossvalidation.py` (purged/embargoed CV) | HKUDS/Vibe-Trading `agent/src/quantlib/crossvalidation.py` | MIT, Copyright (c) 2026 Vibe-Trading Contributors | `7f6908b7e1b95d971f4151bfaf2382e2ce9485cf` | not copied |
| `multipletesting.py` (PSR/DSR/PBO) | HKUDS/Vibe-Trading `agent/src/quantlib/multipletesting.py` | MIT (same) | same | not copied |
| `ledger.py` (hash-chained audit ledger) | HKUDS/Vibe-Trading `agent/src/governance/ledger.py` | MIT (same) | same | not copied |
| small parts of `metrics.py` | HKUDS/Vibe-Trading `agent/backtest/metrics.py` | MIT (same) | same | not copied |

When copied, the full MIT permission notice and copyright line of the origin must be retained alongside the code.

## Dependencies and references

| Project | Role | Licence | Notes |
|---|---|---|---|
| MetaTrader5 (Python package) | Runtime dependency (Windows only), wrapped in `brokers/mt5` | MIT (package metadata, MetaQuotes 2000-2025) | MetaTrader 5 terminal and broker terms are separate |
| Vibe-Trading | Reference; optional read-only research sidecar | MIT | NOTICE says it bundles Qlib feature definitions under Apache 2.0 (unresolved discrepancy with Qlib's MIT `LICENSE`); we do not use its factor zoo |
| Microsoft Qlib | Reference only | MIT (Microsoft Corporation) | no code used |
| Freqtrade / FreqAI | Reference only | **GPL-3.0** (verified: LICENSE is the GPL v3 text) | **No code may be copied into this repository** |

Python dependencies proposed in `docs/IMPLEMENTATION_PLAN.md` have not been licence-audited yet; no GPL/AGPL in the runtime closure without approval.
