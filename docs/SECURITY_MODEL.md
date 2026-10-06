# Security Model

## 1. Assets

Broker credentials (MT5 login/password/server), account identifiers, API keys (calendar, LLM), database credentials, model artefacts and data (proprietary), the ability to place orders.

## 2. Rules

1. **No secrets in git.** `.env` is ignored; `.env.example` has fake placeholders only. Secret scanning (`run_secret_scanning` / gitleaks) in CI and pre-commit (Phase 1).
2. Secrets come from environment variables or an OS/secret manager; never from YAML configs. Config files hold only references (e.g. `metadata_db_url_env`).
3. **Logs never expose secrets:** the logging formatter redacts sensitive keys and `key=value` secrets (`monitoring/logging_setup.py`, tested). Account numbers are treated as sensitive. Exceptions from broker libraries are sanitised before logging.
4. **Demo-first:** development uses demo credentials only. Adapter verifies `account.is_demo` for DEMO mode and refuses otherwise. A live account in DEMO mode (or a demo in LIVE) is a hard error.
5. **LIVE interlock:** `TRADING_MODE=LIVE` + `LIVE_TRADING_ENABLED=true` + matching `LIVE_ACKNOWLEDGEMENT` (human-set; no default), plus runtime checks (broker connected, account verified, symbol verified, risk config loaded, model validated, data healthy, kill switch healthy). Disabled by default; not exercised in Phases 0-13.
6. **Least privilege:** the research agent and dashboard have read-only DB roles; only the execution service has order capability; the dashboard cannot place orders and can only *engage* the kill switch.
7. **Controlled self-improvement only:** no self-modifying production code; models change only via the registry with human-approved promotion; risk limits are human-controlled files.
8. **LLM safety:** agent inputs include untrusted text (news, comments); the agent has no tools that can place orders or alter limits. Agent outputs are advisory.
9. **Supply chain:** pinned dependencies with hashes, minimal dependency set, license check, no vendoring of copyleft code (Freqtrade), review of any MCP/agent sidecar before connecting it to data.
10. **Network:** dashboard bound to localhost/VPN by default; auth required if exposed; TLS for remote DB; collector-engine channel authenticated and local.
11. **Data protection:** disk encryption on the trading host; backups of journal/registry; credentials rotated if ever exposed (a leaked demo password is still rotated).
12. **Windows trading host hardening:** dedicated user, OS updates, firewall, no unrelated software on the MT5 machine, 2FA on broker portal.

## 3. Threats considered

Credential leakage via git/logs/crash dumps; accidental live trading; duplicate or runaway orders (bugs or retries); stale-data trading; compromised dependency; prompt injection through news/text into the agent; unauthorised dashboard access; tampering with risk config or model artefacts (artefact hashes verified on load).

## 4. Open items

Secret-manager choice (D-10), CI secret scanning setup (Phase 1), artefact signing approach, dashboard auth model (Phase 10).
