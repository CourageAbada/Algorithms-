Leakage tests (Phase 3+): every feature must be invariant to truncating future data
(feature(t) computed on data[:t] == feature(t) computed on full data). Labels are the only
future-dependent objects. See docs/VALIDATION_PROTOCOL.md.
