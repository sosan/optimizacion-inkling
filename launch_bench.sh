#!/bin/bash
# launch_bench.sh - run the cross-platform Python benchmark with the repo defaults.
# Equivalent to: python tests/bench_perf.py cli --ctx 344064
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
exec python3 tests/bench_perf.py cli --ctx 344064 "$@"
