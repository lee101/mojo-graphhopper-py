#!/usr/bin/env bash
# Run one thing under the shared bench lock, but yield it as soon as another
# contender is waiting. A naive `flock` here queues for tens of minutes behind
# other repositories' benchmarks on this box; this waits for a free moment,
# runs once, and gives the lock back.
#
# Usage: bench/try-lock.sh <timeout-seconds> <command...>
set -euo pipefail

timeout_s="$1"; shift
lock=/tmp/mojo-bench.lock

deadline=$(( $(date +%s) + timeout_s ))
while :; do
    exec 9>>"$lock"
    if flock -n 9; then
        printf 'lock acquired %s\n' "$(date -u +%FT%TZ)" >&2
        set +e
        "$@"
        rc=$?
        set -e
        printf 'bench exited %d %s\n' "$rc" "$(date -u +%FT%TZ)" >&2
        exit "$rc"
    fi
    exec 9>&-
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "gave up waiting for $lock after ${timeout_s}s" >&2
        exit 75
    fi
    sleep 3
done
