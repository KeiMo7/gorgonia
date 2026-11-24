#!/usr/bin/env bash
#
# Verification script for the TopK / ArgMax / ArgMin selection-op family.
#
# It runs, in order:
#   1. a clean build of the package
#   2. the full TopK-family test suite (verbose)
#   3. the complete package regression suite (to prove existing reduction /
#      autodiff / engine behaviour is unchanged)
#
# All output is teed next to this script in output/ so the run is auditable.
#
# The repository's pinned go4.org/unsafe/assume-no-moving-gc dependency predates
# the installed Go runtime and requires the compatibility env var below; this is
# a pre-existing property of the baseline repository and unrelated to the new
# ops.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Prefer the project toolchain when present.
if [ -x /home/wll/go-sdk/bin/go ]; then
	export PATH=/home/wll/go-sdk/bin:$PATH
fi
export ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.23

OUT_DIR="$ROOT/output"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/topk_verify.log"
: > "$LOG"

run() {
	echo "=== $* ===" | tee -a "$LOG"
	"$@" 2>&1 | tee -a "$LOG"
}

run go version
# Build the main package. (The cmd/cudagen target requires a CUDA toolchain
# that is absent from this environment and already fails to build on the
# untouched baseline, so it is intentionally not part of this run.)
run go build .
run go test -count=1 -v -run 'TopK|ArgMax|ArgMin' .
run go test -count=1 .

echo "All verification steps passed. Log saved to $LOG" | tee -a "$LOG"
