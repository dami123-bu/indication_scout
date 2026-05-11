"""Test-only constants for the regression suite (paths + cassette mode env).

Numeric thresholds for the comparison harness live in
`indication_scout.regression.constants` so they're importable from both the
test suite and the `scout diff-report` CLI.
"""

from pathlib import Path

REGRESSION_DIR = Path(__file__).parent
GOLDEN_DIR = REGRESSION_DIR / "golden"
CASSETTE_DIR = REGRESSION_DIR / "cassettes"

CASSETTE_MODE_ENV = "SCOUT_CASSETTE_MODE"
CASSETTE_MODE_REPLAY = "replay"
CASSETTE_MODE_RECORD = "record"
CASSETTE_MODE_LIVE = "live"
