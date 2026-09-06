#!/usr/bin/env bash
# reproduce.sh: validate and refresh the frozen study. Tracked splits and results are skipped,
# because results are bound to split content by sha256 and rebuilding a split would invalidate
# every finished run. Step 7 is the training programme, about five hours on an Apple M-series
# laptop; the other steps take minutes and the analyses always rerun. Interrupting step 1 or 3
# can leave a partial archive or cache that looks complete: delete the artifact in flight before
# re-running.
set -uo pipefail
die () { echo "STEP FAILED: $*" >&2; exit 1; }

# Same default as run_all.sh, so the audits and the training run under one interpreter.
# ENVIRONMENT. This expects a virtual environment built from requirements.txt with the pinned
# versions. If it is absent, set PY to an interpreter that has those packages.
if [ -x "../.venv/bin/python" ]; then PY="${PY:-../.venv/bin/python}"
else PY="${PY:-python3}"
     echo "WARNING: .venv missing, falling back to python3 on PATH." >&2
     echo "         rebuild it with: python3 -m venv ../.venv && \\" >&2
     echo "         ../.venv/bin/pip install -r requirements.txt" >&2
fi
CAMS="858,3888,4795,5021,1093,3297,4232,4801,19106,8438,17218,9730"

echo "== 1. metadata table and images ============================"
"$PY" data/download.py || die "data/download.py"

echo; echo "== 2. labels ==============================================="
[ -f data/labels.csv ] || "$PY" data/build_labels.py --csv data/complete_table_with_mcr.csv \
    --out data/labels.csv --require-zip || die "data/build_labels.py"

echo; echo "== 3. decode cache, dropping undecodable files ============="
# The cache is not tracked, so on a fresh clone this always runs. The tracked splits are not
# rebuilt, so the rebuilt index is compared against the tracked label file the splits were built
# from, and any difference stops the run.
if [ ! -f data/cache_112.npy ]; then
  "$PY" data/build_cache.py || die "data/build_cache.py"
  if [ -f data/labels_cached.csv ]; then
    cmp -s data/cache_112_index.csv data/labels_cached.csv || die \
      "the rebuilt cache index differs from the tracked data/labels_cached.csv. The stored results
       are bound to the tracked splits, which were built from that file, so this cache does not hold
       the images the study used. Compare data/cache_112_index.csv against it before continuing."
    echo "  rebuilt cache index matches the tracked label file"
  else
    cp -f data/cache_112_index.csv data/labels_cached.csv || die "cache index copy"
  fi
else
  echo "  cache present, skipping (delete data/cache_112.npy to force a rebuild)"
fi

echo; echo "== 4. splits, both regimes ================================="
# Skipped when present for the same reason as step 3: results are bound to split content.
if [ ! -f data/splits_iid.csv ]; then
  "$PY" data/build_splits.py --labels data/labels_cached.csv --out data/splits_iid.csv \
      --test-cap 3 --val-cap 4 || die "data/build_splits.py iid"
  for f in 0 1 2 3; do
    "$PY" data/build_splits.py --labels data/labels_cached.csv --regime camera-disjoint \
        --fold $f --out data/splits_cd$f.csv || die "data/build_splits.py cd$f"
  done
else
  echo "  splits present, skipping (delete data/splits_iid.csv to force a rebuild)"
fi

echo; echo "== 5. no-pixel controls, both regimes ======================"
"$PY" baselines/trivial.py --splits data/splits_iid.csv || die "baselines/trivial.py"
for f in 0 1 2 3; do
  "$PY" baselines/trivial.py --splits data/splits_cd$f.csv || die "baselines/trivial.py cd$f"
done
"$PY" baselines/trivial.py --fold-means || die "baselines/trivial.py --fold-means"

echo; echo "== 6. audits: label timing, porting bugs, leakage =========="
"$PY" audit/label_timing.py --cameras "$CAMS" || die "audit/label_timing.py"
"$PY" dir/porting_bugs.py || die "dir/porting_bugs.py"
# 4.535 is the in-distribution LOOKUP MAE printed by step 5; P3 compares the nearest-frame oracle
# against it, and the two must be updated together.
"$PY" probes/leakage.py --splits data/splits_iid.csv --control-mae 4.535 || die "probes/leakage.py iid"
"$PY" probes/leakage.py --splits data/splits_cd0.csv || die "probes/leakage.py"

echo; echo "== 7. training ============================================="
./run_all.sh || die "run_all.sh: the table and report steps must not run on a partial results/ directory"

echo; echo "== 8. tables and result integrity ========================="
n=$(ls results/*.json 2>/dev/null | wc -l | tr -d ' ')
if [ "$n" -lt 32 ]; then
  echo "  only $n/32 training results present; fill_report.py writes the tables only when every arm matches ERM's seed count, and each arm carries its seed count in brackets"
fi
"$PY" analysis/paired_arms.py --a erm --b sqinv || die "analysis/paired_arms.py"
"$PY" analysis/paired_arms.py --a sqinv --b lds || die "analysis/paired_arms.py"
"$PY" analysis/paired_arms.py --a erm --b lds || die "analysis/paired_arms.py"
for f in 0 1 2 3; do
  "$PY" analysis/paired_arms.py --a sqinv --b lds --prefix cd$f --splits data/splits_cd$f.csv \
      || die "analysis/paired_arms.py cd$f"
done
"$PY" dir/test_fds_last_epoch.py || die "dir/test_fds_last_epoch.py"
"$PY" fill_report.py           || die "fill_report.py"
"$PY" check_results.py
