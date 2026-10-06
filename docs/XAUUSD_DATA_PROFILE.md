# XAU/USD Data Profile

**Status: NOT YET POPULATED. No real XAU/USD data has been collected.**
Phase 1 was built and tested in a Linux environment where MetaTrader 5 cannot run. This file will be generated from real
broker data after the operator runs the Windows procedure (`docs/WINDOWS_MT5_VERIFICATION.md`, step 7). Until then it
contains **no measurements and no conclusions**; any number appearing here before that run would be fabricated.

## How this file is produced
```powershell
python -m scripts.profile_ticks --broker "<company>" --server "<server>" --instrument XAU_USD `
    --start <YYYY-MM-DD> --end <YYYY-MM-DD> --output docs\XAUUSD_DATA_PROFILE.md --json-output reports\xauusd_profile.json
```
The generator (`src/fxscalp/market_data/profile.py`) reads only verified stored chunks and writes the sections below.
It is **descriptive data analysis only**: no signal, strategy, or performance statement is produced (a test enforces that
the output contains no performance vocabulary). Quarantined ticks are excluded from spread/volatility statistics but are
always counted and reported.

## Sections that will be filled (definitions)
| Section | What it contains | Notes |
|---|---|---|
| Coverage | chunks present/missing/empty days, tick counts, quarantined ticks | missing vs empty are distinct |
| Tick frequency | ticks per day, mean ticks per UTC hour, ticks by session, inter-tick interval percentiles | session labels from `sessions/calendar.py` (DST-aware conventions, to be validated here) |
| Spread | percentiles (p1..p99.9) of spread in points and relative spread; by UTC hour; by session; zero-spread count; widest ticks | points = (ask-bid)/point, with `point` from the stored symbol metadata |
| Missing data / duplicates | counts per quality flag; long gaps | flags: negative/zero spread, missing bid/ask, duplicate timestamp/tick, reversal, large gap, sec/msec mismatch, ambiguous time |
| Weekend boundaries | last tick before and first tick after each weekend, in New York time, with gap length | tests the "weekly open ~ 17:00 New York" assumption used for DST inference |
| Rollover | spread percentiles by minute offset from 17:00 New York | descriptive |
| Volatility | std and mean absolute 1-minute mid return (bps) by UTC hour and session | descriptive; not a forecast |
| Broker symbol metadata | all fields as returned, with units unassumed, plus consistency findings | from `data/metadata/symbol_info/...` |

## Open questions this profile is meant to answer (not answered yet)
1. How many ticks per day does this broker deliver for gold, and how does it vary by hour/session?
2. What does the spread distribution look like per session, and how wide are the tails?
3. How do spreads behave around the daily rollover and at the weekly open/close?
4. How much missing/duplicated/reversed data is there, and where?
5. Do the session conventions in `sessions/calendar.py` match observed activity?
