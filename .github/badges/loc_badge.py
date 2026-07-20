"""Regenerate the local lines-of-code badge JSON.

Extended Summary
----------------
Counts logical lines of code in ``src/diffpes`` with pygount and
writes ``.github/badges/loc.json`` in the shields.io endpoint schema.
Runs as a local pre-commit hook so the badge updates inside normal
commits — no CI job ever commits to the repository for this badge.
The file is rewritten only when the count changes, keeping the hook
silent on unrelated commits.

Routine Listings
----------------
:func:`main`
    Count lines of code and rewrite the badge JSON if it changed.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
_BADGE_PATH: Path = _REPO_ROOT / ".github" / "badges" / "loc.json"
_COUNT_TARGET: Path = _REPO_ROOT / "src" / "diffpes"


def main() -> int:
    """Count lines of code and rewrite the badge JSON if it changed.

    Returns
    -------
    exit_code : int
        Zero on success (pre-commit detects the modified badge file
        itself); one if pygount is unavailable or its output cannot
        be parsed.
    """
    pygount = Path(sys.executable).parent / "pygount"
    result = subprocess.run(  # noqa: S603
        [
            str(pygount),
            "--format=cloc-xml",
            "--suffix=py",
            str(_COUNT_TARGET),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(f"pygount failed: {result.stderr.strip()}", file=sys.stderr)
        return 1
    match = re.search(r'<total[^>]*\bcode="(\d+)"', result.stdout)
    if match is None:
        print("could not parse pygount cloc-xml total", file=sys.stderr)
        return 1
    loc: str = match.group(1)
    badge: str = (
        json.dumps(
            {
                "schemaVersion": 1,
                "label": "lines of code",
                "message": loc,
                "color": "blue",
            },
            indent=2,
        )
        + "\n"
    )
    if _BADGE_PATH.exists() and _BADGE_PATH.read_text() == badge:
        return 0
    _BADGE_PATH.write_text(badge)
    print(f"updated {_BADGE_PATH.relative_to(_REPO_ROOT)}: {loc} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
