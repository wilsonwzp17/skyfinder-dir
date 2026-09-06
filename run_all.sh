#!/usr/bin/env bash
# run_all.sh - the in-distribution ablation and the camera-held-out folds, in one process.
# Safe to re-run: a result is skipped when check_results.py accepts it, regenerated otherwise,
# and each result is written atomically, so a killed run loses at most the run in flight.
set -uo pipefail
# ENVIRONMENT. This expects a virtual environment built from requirements.txt with the pinned
# versions. If it is absent, set PY to an interpreter that has those packages.
if [ -x "../.venv/bin/python" ]; then PY="${PY:-../.venv/bin/python}"
else PY="${PY:-python3}"
     echo "WARNING: .venv missing, falling back to python3 on PATH." >&2
     echo "         rebuild it with: python3 -m venv ../.venv && \\" >&2
     echo "         ../.venv/bin/pip install -r requirements.txt" >&2
fi

run () {  # splits arm seed outfile
  if "$PY" check_results.py --file "$4" >/dev/null 2>&1; then
    echo "SKIP $(basename "$4") (valid result exists)"; return
  fi
  rm -f "$4" "$4.partial"
  echo "=== $(basename "$4" .json) ==="
  "$PY" dir/train.py --splits "$1" --arm "$2" --seed "$3" --basis bin --epochs 40 \
      --reweight sqrt_inv --out "$4" 2>&1 | tail -14 \
    || { echo "training failed for $(basename "$4"); stopping" >&2; exit 1; }
}

ALL=32

# NOSMOOTH GATE. Called below once both inputs exist; on a cold start neither file exists before
# the loop.
nosmooth_gate () {
  [ -f results/iid_fdsNOSMOOTH_s0.json ] && [ -f results/iid_erm_s0.json ] || {
    echo "NOSMOOTH gate cannot run: the diagnostic or same-seed ERM is missing" >&2; exit 1; }
  "$PY" - <<'GATE' || { echo "NOSMOOTH gate failed; not launching FDS arms" >&2; exit 1; }
import json,sys
a=json.load(open("results/iid_fdsNOSMOOTH_s0.json"))["errors"]
b=json.load(open("results/iid_erm_s0.json"))["errors"]
sys.exit(0 if a==b else 1)
GATE
  echo "NOSMOOTH gate passed: the diagnostic is bit-identical to same-seed ERM"
}

# Up to three passes: a result that the manifest rejects after the loop has walked past it is
# regenerated on the next pass instead of needing a manual restart.
for PASS in 1 2 3; do
n_valid () { "$PY" check_results.py 2>/dev/null | awk '/valid,/{print $1}'; }
have=$(n_valid); have=${have:-0}
if [ "$have" -ge "$ALL" ] && "$PY" check_results.py >/dev/null 2>&1; then
  echo "########## ALL $ALL RESULTS PRESENT AND VALID ##########"; break
fi
echo "########## PASS $PASS  ($have/$ALL valid) ##########"

echo "########## E1  FDS diagnostic, run FIRST ##########"
# E1 first. Does the FDS collection pass alone change the predictions? It must not: with
# start_smooth beyond the epoch count smooth() never fires, so this arm has to equal same-seed ERM
# bit for bit before any FDS-bearing arm runs; otherwise the ten FDS-bearing results, about three
# hours on this machine, could not be reported. A match shows only that the collection pass is
# inert, because the inherited FDS state behaviour begins when smoothing does.
if "$PY" check_results.py --file results/iid_fdsNOSMOOTH_s0.json >/dev/null 2>&1; then
  echo "SKIP iid_fdsNOSMOOTH_s0 (valid result exists)"
else
  rm -f results/iid_fdsNOSMOOTH_s0.json results/iid_fdsNOSMOOTH_s0.json.partial
  echo "=== iid_fdsNOSMOOTH_s0 ==="
  "$PY" dir/train.py --splits data/splits_iid.csv --arm fds --seed 0 --basis bin --epochs 40 \
      --reweight sqrt_inv --start-smooth 999 --out results/iid_fdsNOSMOOTH_s0.json 2>&1 | tail -14 \
    || { echo "the no-smoothing diagnostic failed; stopping" >&2; exit 1; }
fi

echo "########## E2  in-distribution ablation ##########"
# ERM, SQINV and SQINV + LDS first: none touches the FDS collection path, and ERM seed 0 is the
# gate's input.
for arm in erm sqinv lds; do
  for s in 0 1 2; do run data/splits_iid.csv $arm $s "results/iid_${arm}_s${s}.json"; done
done

# Both gate inputs now exist. Nothing FDS-bearing runs until they agree exactly.
nosmooth_gate

for arm in fds lds_fds; do
  for s in 0 1 2; do run data/splits_iid.csv $arm $s "results/iid_${arm}_s${s}.json"; done
done

echo "########## E3  camera held out ##########"
nosmooth_gate
for f in 0 1 2 3; do
  for arm in erm sqinv lds lds_fds; do run data/splits_cd${f}.csv $arm 0 "results/cd${f}_${arm}_s0.json"; done
done

done
echo "########## ALL DONE ##########"
"$PY" check_results.py
