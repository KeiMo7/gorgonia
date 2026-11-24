#!/usr/bin/env bash
#
# Verification script for the differentiation lifecycle / gradient buffer /
# gradient check work.
#
# It runs, in order:
#   1. a clean build of the package
#   2. the new lifecycle, pool, check and training-session tests (verbose)
#   3. the same tests under the race detector
#   4. the complete package regression suite (to prove existing autodiff,
#      solver, machine and graph behaviour is unchanged)
#
# Output is teed next to this script so the run is auditable.
#
# The repository's pinned go4.org/unsafe/assume-no-moving-gc dependency predates
# the installed Go runtime and requires the compatibility env var below; this is
# a pre-existing property of the baseline repository and unrelated to this work.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Prefer the project toolchain when present.
if [ -x /home/wll/go-sdk/bin/go ]; then
	export PATH=/home/wll/go-sdk/bin:$PATH
fi
# The guard compares against runtime.Version() trimmed to "goMAJOR.MINOR".
GO_MM="$(go env GOVERSION | awk -F. '{print $1 "." $2}')"
export ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH="$GO_MM"

OUT_DIR="$ROOT/output"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/lifecycle_verify.log"
: > "$LOG"

SELECT='TestDifferentiate|TestGradCheck|TestGradientPool|TestTrainingSession|TestMarkGradDirty'

run() {
	echo "=== $* ===" | tee -a "$LOG"
	"$@" 2>&1 | tee -a "$LOG"
}

run go version
run go build .
run go test -count=1 -v -run "$SELECT" .
run go test -count=1 -race -run "$SELECT" .
run go test -count=1 .

echo "All verification steps passed. Log saved to $LOG" | tee -a "$LOG"
