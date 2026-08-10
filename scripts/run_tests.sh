#!/usr/bin/env bash
#
# Run the test suite.
#
#   scripts/run_tests.sh                  # whole suite, quiet
#   scripts/run_tests.sh -v               # verbose, one line per test
#   scripts/run_tests.sh --warnings       # keep third-party DeprecationWarnings
#   scripts/run_tests.sh tests/test_physics.py -k sod    # pass anything through to pytest
#
# Third-party warnings (warp, torch on py3.14) are suppressed by default --
# they bury the actual result. --warnings brings them back.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PYTEST_ARGS=()
SHOW_WARNINGS=0
VERBOSE=0

for arg in "$@"; do
    case "$arg" in
        --warnings) SHOW_WARNINGS=1 ;;
        -v|--verbose) VERBOSE=1 ;;
        *) PYTEST_ARGS+=("$arg") ;;
    esac
done

CMD=(python -m pytest)
[[ $SHOW_WARNINGS -eq 0 ]] && CMD+=(-p no:warnings)
if [[ $VERBOSE -eq 1 ]]; then
    CMD+=(-v)
else
    CMD+=(-q)
fi
CMD+=("${PYTEST_ARGS[@]}")

echo "== warpSPH test suite =="
echo "\$ ${CMD[*]}"
echo

# Tests bootstrap float32 in tests/conftest.py; nothing to set up here.
if "${CMD[@]}"; then
    echo
    echo "PASS"
else
    status=$?
    echo
    echo "FAIL (pytest exit $status)"
    exit $status
fi
