#!/bin/bash
# Pull (seed, runs) jobs from a shared job file and run them one at a time.
#   scripts/knit_queue_worker.sh JOBFILE THREADS [extra rerun_sec5.py args...]
# Job file lines: "<seed> <run indices>", e.g. "20261126 0 1 2 3" or "20261126 5".
# Several workers may share one job file (flock); PY selects the interpreter.
JOBS=$1; THREADS=$2; shift 2
PY=${PY:-python3}
while true; do
  job=$(flock "$JOBS.lock" bash -c "head -n1 '$JOBS'; sed -i 1d '$JOBS'")
  [ -z "$job" ] && { echo "$(date) queue empty"; break; }
  seed=${job%% *}; runs=${job#* }
  echo "$(date) start seed $seed runs $runs"
  $PY -u scripts/rerun_sec5.py --stage knitted --seeds $seed --runs $runs --threads $THREADS "$@"
  echo "$(date) done seed $seed runs $runs (exit $?)"
done
