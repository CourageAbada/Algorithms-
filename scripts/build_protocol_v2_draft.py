"""Generate (and statically validate) the Protocol v2 DRAFT configuration files. Trains nothing; activates nothing."""

from __future__ import annotations

import sys

from fxscalp.research.protocol_v2 import draft


def main() -> int:
    idx = draft.write_draft()
    problems = draft.validate_draft()
    print(f"draft_hash {idx['draft_hash']}; {len(idx['files'])} files; status {idx['status']}")
    if problems:
        print("VALIDATION PROBLEMS:\n  " + "\n  ".join(problems))
        return 1
    print("static validation: OK (draft remains NOT ACTIVE)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
