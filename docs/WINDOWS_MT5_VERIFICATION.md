# Windows MT5 Verification Procedure (Phase 1)

**Purpose:** run the read-only verification harness against a real MetaTrader 5 DEMO terminal on Windows, collect the
JSON report, and (optionally) acquire and profile a first sample of XAU/USD ticks.

**Why this is needed:** the `MetaTrader5` Python package is Windows-only. Everything in Phase 1 was built and tested on
Linux against a fake; **nothing about real MT5 behaviour is verified until you complete this procedure and send back
the report.** Nothing in this procedure sends an order, and the code cannot call any order function.

**Never** put credentials in source files, commit `.env`, or paste passwords/account numbers into chats or tickets.

---

## 0. What you need
- Windows 10/11 x64 (not ARM), admin rights to install software.
- A broker with MT5 and a **DEMO** account (login number, password, server name). Gold must be offered (any of
  `XAUUSD`, `XAUUSDm`, `GOLD`, ... - the code discovers it).
- Run the main verification **while gold is trading** (Sunday evening to Friday evening New York time). When the market
  is closed the time calibration is inconclusive (by design).

## 1. Install Python (64-bit)
1. Download Python 3.12 x64 from https://www.python.org/downloads/windows/ and install (tick **Add python.exe to PATH**).
2. In PowerShell:
   ```powershell
   python --version
   python -c "import struct,platform;print(struct.calcsize('P')*8, platform.machine())"
   ```
   Expected: `Python 3.12.x` and `64 AMD64`. If you see `32` or `ARM64`, reinstall the x64 build.

## 2. Get the code and create the virtual environment
```powershell
git clone https://github.com/CourageAbada/Algorithms-.git
cd Algorithms-
git checkout claude/ai-forex-scalping-platform-vz1h0z
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass      # only affects this window
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,mt5]"
```
`[mt5]` installs the official `MetaTrader5` package (Windows only).
```powershell
pip show MetaTrader5          # record the version in your notes (reference version: 5.0.6231)
```

## 3. Run the unit tests (no MT5 needed)
```powershell
python -m pytest -q
```
Expected: all pass. If anything fails here, stop and send the output.

## 4. Install the MT5 terminal and log in to the DEMO account
1. Install the MetaTrader 5 terminal from your broker (or from metatrader5.com) and start it.
2. File > Open an Account / Login to Trade Account: enter the **demo** login, password and server.
3. Confirm the bottom-right connection indicator shows data flowing (not "No connection").
4. In Market Watch right-click > Symbols (or "Show All") and make sure the gold symbol is visible. (The harness also
   adds symbols to Market Watch itself; that is data visibility, not trading.)
5. Leave the terminal **running and logged in** during every step below.
6. Tools > Options > Charts: set **Max bars in chart** high (e.g. 1000000) if you want deep bar history (affects bars,
   not the tick requests).
7. Windows clock: `w32tm /query /status` should show a recent sync; if unsure run `w32tm /resync` (as admin). The
   time-semantics measurement assumes the local clock is accurate to a few seconds.

## 5. Configure `.env` (never commit it)
```powershell
copy .env.example .env
notepad .env
```
Set (values are examples):
```
MT5_LOGIN=<your demo account number>
MT5_PASSWORD=<your demo password>
MT5_SERVER=<exact server name shown in the terminal, e.g. BrokerName-Demo>
MT5_TERMINAL_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
```
If the terminal is already logged in you may leave login/password unset (the adapter attaches to the running terminal),
but set `MT5_TERMINAL_PATH` if you have more than one MT5 installation.
Check the file is ignored by git: `git status` must **not** list `.env`.

## 6. Run the verification
```powershell
python -m scripts.verify_mt5 --time-samples 60 --time-interval 2 --dst-weeks 52 --tick-hours 1
```
Flags: `--time-samples/--time-interval` offset sampling (60 x 2 s); `--dst-weeks 52` probes the weekly open for the last 52
weeks (one cheap call per week) to infer DST behaviour; `--skip-dom` skips the depth-of-market probe; `--no-probe` skips
the history-depth/request-limit probes. It takes a few minutes.

Output: lines `[PASS|WARN|FAIL|SKIP] id: title`, then `Verdict:` and the path of the JSON report in `reports\`
(`mt5_verify_<UTC timestamp>.json`). The report contains: environment, masked login, server, account class, terminal
info, symbol discovery per instrument, **raw and typed symbol metadata**, latest ticks, the **time-semantics
measurement**, a tick sample with quality analysis, bar sample, DOM probe, history-depth and request-limit probes,
latency, a no-trading attestation, and the list of still-unverified items. It does **not** contain your password.

Exit codes: `0` ok, `1` a check failed, `2` connection/initialisation problem, `3` account not DEMO (nothing collected),
`4` XAU/USD could not be resolved.

Verify the report is clean before sharing:
```powershell
Select-String -Path reports\*.json -Pattern "<your password>","<your full login number>"
```
This must print nothing.

## 7. (Recommended) collect a first XAU/USD sample and profile it
Pick the last 5 trading days (use the calendar dates of the broker's server time):
```powershell
python -m scripts.acquire_ticks --instrument XAU_USD --start 2026-09-28 --end 2026-10-02
python -m scripts.acquire_ticks --instrument XAU_USD --start 2026-09-28 --end 2026-10-02   # run again: all chunks "skipped_verified"
python -m scripts.profile_ticks --broker "<company name from the report>" --server "<server from the report>" `
    --instrument XAU_USD --start 2026-09-28 --end 2026-10-02 --output docs\XAUUSD_DATA_PROFILE.md --json-output reports\xauusd_profile.json
```
The acquirer stops with an explicit message if the time basis is `unverified`; fix the calibration (market open,
`--calibrate`) rather than forcing it, unless you deliberately accept flagged data with `--allow-unverified-time`.

## 8. Optional benchmarks on the Windows machine
```powershell
python -m scripts.benchmark_io --rows 2000000 --repeat 3 --json-output reports\windows_synthetic_io.json
```
(Synthetic parquet/convert baselines for this machine. MT5 retrieval throughput is already in the verify report under
`ticks.rows_per_s` and `latency`.)

## 9. What to send back
- `reports\mt5_verify_*.json` (all of them if you ran more than one)
- `data\metadata\timebase\...\*.json` (time calibration spec)
- `configs\broker_symbols\*.yaml` (symbol map)
- `data\metadata\symbol_info\...\*.json` (symbol metadata snapshots)
- `data\acquisition_log.jsonl`, the dataset manifest(s) in `data\datasets\...`, and `docs\XAUUSD_DATA_PROFILE.md` if step 7 was run
- the terminal build number and your Windows version
**Do not send `.env`, screenshots showing your login/password, or the raw tick parquet files (large).**

## 10. Troubleshooting
| Symptom | Likely cause / action |
|---|---|
| `MetaTrader5 is not installed` | `pip install MetaTrader5` inside the activated venv (64-bit Python only) |
| `requires 64-bit Python` / `DLL load failed` | 32-bit Python or ARM Windows; install x64 Python |
| `MT5 initialize() failed (-10003 ...)` | Terminal not running, or wrong `MT5_TERMINAL_PATH`; start the terminal and log in |
| `MT5 login failed (-6 ...)` | Wrong login/password/server for the DEMO account |
| `AccountNotAllowedError ... REAL` (exit 3) | The terminal is logged into a real/contest account. Log into a DEMO account. No data was collected |
| `AccountMismatchError` | `.env` login/server differ from the terminal's current account |
| time_offset FAIL, "tick did not advance" | Market closed or illiquid; run during active hours; check Windows time sync |
| time_offset FAIL, "residual ... clock" | Local clock is off; `w32tm /resync` |
| `NoTickDataError` | Symbol has no history yet; open a chart of the symbol and wait for download, then retry |
| `CallTimeoutError` | Terminal hung; restart MT5 and retry |
