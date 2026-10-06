import json
import logging

from fxscalp.monitoring.logging_setup import REDACTED, JsonFormatter, redact


def test_redacts_sensitive_keys_recursively():
    out = redact({"a": 1, "MT5_PASSWORD": "hunter2", "n": {"api_key": "k", "ok": "v"}})
    assert out["MT5_PASSWORD"] == REDACTED
    assert out["n"]["api_key"] == REDACTED
    assert out["n"]["ok"] == "v" and out["a"] == 1


def test_redacts_inline_secrets_in_messages():
    rec = logging.LogRecord("t", logging.INFO, "f", 1, "login failed password=hunter2 x", None, None)
    payload = json.loads(JsonFormatter().format(rec))
    assert "hunter2" not in payload["msg"]
