#!/usr/bin/env python3
"""Validate every result file against the manifest of the study.

Usage:  python3 check_results.py [--dir results] [--file results/x.json]
"""
import argparse, collections, csv, glob, hashlib, json, os, sys

REQUIRED = ["arm", "seed", "splits", "table", "errors", "n_train", "n_val", "n_test",
            "best_epoch", "epochs", "batch", "lr"]
COLS = ["Overall", "Many", "Medium", "Few", "Zero"]


# Every result file the study should produce.
EXPECTED = (["results/iid_fdsNOSMOOTH_s0.json"]
            + [f"results/iid_{a}_s{s}.json"
               for a in ("erm", "sqinv", "lds", "fds", "lds_fds") for s in range(3)]
            + [f"results/cd{f}_{a}_s0.json" for f in range(4) for a in ("erm", "sqinv", "lds", "lds_fds")]
            )


def nosmooth_gate():
    """The no-smoothing control collects FDS statistics without applying them, so its residuals
    must match same-seed ERM exactly."""
    a, b = "results/iid_fdsNOSMOOTH_s0.json", "results/iid_erm_s0.json"
    if not (os.path.exists(a) and os.path.exists(b)):
        return "the no-smoothing control or its ERM counterpart is missing, so the gate cannot run"
    ea, eb = json.load(open(a))["errors"], json.load(open(b))["errors"]
    if len(ea) != len(eb): return f"length mismatch {len(ea)} vs {len(eb)}"
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in ea + eb):
        return "one of the two files has a non-numeric residual, so the gate cannot run"
    d = max(abs(x - y) for x, y in zip(ea, eb))
    return None if d <= 1e-9 else f"differs from same-seed ERM by up to {d:.3e}"


# The configuration each expected file must carry. `basis` is the shot-region definition the
# result was scored on; the report recomputes every cell on marginal bins, so every run stores bin.
EXPECTED_CFG = {}
# start_smooth beyond the epoch count is what makes this the no-smoothing control.
EXPECTED_CFG["results/iid_fdsNOSMOOTH_s0.json"] = dict(
    arm="fds", seed=0, splits="data/splits_iid.csv", epochs=40, batch=64, lr=0.05, basis="bin",
    reweight="none", start_smooth=999)   # an FDS arm applies no target reweighting
for _a in ("erm", "sqinv", "lds", "fds", "lds_fds"):
    for _s in range(3):
        EXPECTED_CFG[f"results/iid_{_a}_s{_s}.json"] = dict(
            arm=_a, seed=_s, splits="data/splits_iid.csv", epochs=40, batch=64, lr=0.05,
            basis="bin", start_smooth=1,
            reweight=("sqrt_inv" if _a in ("sqinv", "lds", "lds_fds") else None))
for _f in range(4):
    for _a in ("erm", "sqinv", "lds", "lds_fds"):
        EXPECTED_CFG[f"results/cd{_f}_{_a}_s0.json"] = dict(
            arm=_a, seed=0, splits=f"data/splits_cd{_f}.csv", epochs=40, batch=64, lr=0.05,
            basis="bin", start_smooth=1,
            reweight=("sqrt_inv" if _a in ("sqinv", "lds", "lds_fds") else None))

def config_violations(path, d):
    """Check the stored result against its expected configuration."""
    want = EXPECTED_CFG.get(path)
    if want is None:
        return [f"{path}: not in the manifest; unexpected result files are a failure, not a pass"]
    bad = []
    for k, v in want.items():
        got = d.get(k)
        if k == "splits" and got is not None:
            got = os.path.basename(str(got)); v = os.path.basename(v)
        if k == "reweight" and v is None and got in (None, "none"):
            continue
        if got != v:
            bad.append(f"{path}: {k} is {got!r}, manifest says {v!r}")
    return bad


def missing():
    return [f for f in EXPECTED if not os.path.exists(f)]


def split_sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None


def validate(path):
    try:
        d = json.load(open(path))
    except Exception as e:
        return f"unparseable: {type(e).__name__}"
    for k in REQUIRED:
        if k not in d: return f"missing field '{k}'"
    t = d["table"]
    if "Overall" not in t or t["Overall"] is None:
        return "table has no Overall"
    if not isinstance(d["errors"], list) or len(d["errors"]) != d["n_test"]:
        return f"errors length {len(d.get('errors', []))} != n_test {d['n_test']}"
    # Every split row must be in the cache: a count that differs from the split file means the
    # result was scored on a different population.
    sp = d.get("splits")
    if sp and os.path.exists(sp):
        seen = collections.Counter(r["split"] for r in csv.DictReader(open(sp)))
        for field, key in (("n_train", "train"), ("n_val", "val"), ("n_test", "test")):
            if field in d and d[field] != seen[key]:
                return (f"{field} is {d[field]} but {sp} has {seen[key]} {key} rows: "
                        f"this result was scored on a different population")
    for c in COLS:
        if c in t and t[c] is not None and (not isinstance(t[c], list) or len(t[c]) < 3):
            return f"malformed cell '{c}'"
    # A killed run once left a truncated file that satisfied [ -f ]; residuals are therefore
    # checked as finite numbers, not merely present. bool is a subclass of int.
    bad = [i for i, e in enumerate(d["errors"])
           if not isinstance(e, (int, float)) or isinstance(e, bool) or e != e or abs(e) == float("inf")]
    if bad:
        return f"{len(bad)} residuals are not finite numbers, first at index {bad[0]}"
    if not (0 <= d["best_epoch"] < d["epochs"]):
        return f"best_epoch {d['best_epoch']} outside [0, {d['epochs']})"
    # Every stored table cell must agree with a fresh one recomputed from the residuals.
    try:
        import common as _c
        rows = list(csv.DictReader(open(d["splits"])))
        tr = [r for r in rows if r["split"] == "train"]
        te = [r for r in rows if r["split"] == "test"]
        want = _c.dsr_table(d["errors"], te, _c.shot_fn(tr, _c.BIN_WIDTH,
                                                       basis=d.get("basis", "bin"))[0])
    except Exception as e:
        return f"cannot recompute the table: {e}"
    for k, v in want.items():
        if k.startswith("_") or k not in t: continue   # skip private and unstored keys
        got = t.get(k)
        if v is None and got is None: continue
        if (v is None) != (got is None):
            return f"table cell {k!r} is {got!r}, recomputes to {v!r}"
        for i in range(min(len(v), len(got))):
            if isinstance(v[i], (int, float)) and isinstance(got[i], (int, float)):
                if abs(got[i] - v[i]) > 5e-3:
                    return f"table {k}[{i}] is {got[i]}, recomputes to {v[i]:.6f}"
            elif got[i] != v[i]:
                return f"table {k}[{i}] is {got[i]!r}, recomputes to {v[i]!r}"
    # An FDS arm must actually smooth; the control must not.
    arm, ss = d["arm"], d.get("start_smooth")
    if "fds" in arm:
        diag = os.path.basename(path).startswith("iid_fdsNOSMOOTH")
        if diag and (ss is None or ss <= d["epochs"]):
            return f"the no-smoothing diagnostic has start_smooth={ss}, which would smooth"
        if not diag and (ss is None or ss > d["epochs"]):
            return f"FDS arm with start_smooth={ss}: smoothing never fires, this is not an FDS run"
        bs, bn = d.get("fds_bucket_start"), d.get("fds_bucket_num")
        if not isinstance(bs, int) or not isinstance(bn, int) or bn - bs < 2:
            return f"implausible FDS bucket range start={bs} num={bn}"
    if d["arm"] in ("lds", "lds_fds") and d.get("reweight") not in ("sqrt_inv", "inverse"):
        return f"LDS arm with reweight={d.get('reweight')!r}: expected sqrt_inv or inverse"
    # The result must still be bound to the split file it was produced from.
    sha = d.get("split_sha256")
    if not sha:
        return "no split_sha256: not bound to the split it was produced from"
    cur = split_sha(d["splits"])
    if cur is None:
        return f"names split {d['splits']} which no longer exists"
    if cur != sha:
        return (f"split content changed since this result was produced "
                f"({sha[:12]} then, {cur[:12]} now): it is not comparable to fresh results")
    # fds_bn_frozen is the trainer's declaration that common.frozen_bn wrapped the collection
    # pass; the gate on iid_fdsNOSMOOTH_s0 is the test of it, for seed 0.
    if d["arm"] in ("fds", "lds_fds") and not d.get("fds_bn_frozen"):
        return ("FDS arm without fds_bn_frozen: the collection pass would re-estimate the "
                "BatchNorm running statistics")
    if d["arm"] in ("fds", "lds_fds") and d.get("fds_bucket_start") in (None, 0):
        return (f"FDS arm with fds_bucket_start={d.get('fds_bucket_start')!r}: the bucket range "
                f"must start at the first occupied bin")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", default=None)
    ap.add_argument("--file", default=None, help="validate exactly one file and exit")
    a = ap.parse_args()
    if a.file:
        err = validate(a.file) if os.path.exists(a.file) else "does not exist"
        cfg = []
        if not err and os.path.exists(a.file):
            cfg = config_violations(a.file, json.load(open(a.file)))
        if err: print(f"  BAD  {a.file}: {err}")
        for m in cfg: print(f"  CONFIG {m}")
        gate = nosmooth_gate() if os.path.basename(a.file).startswith("iid_fdsNOSMOOTH") else None
        if gate: print(f"  GATE {gate}")
        sys.exit(1 if (err or cfg or gate) else 0)
    dirs = a.dir or ["results"]
    bad, ok, stray = [], 0, []
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "*.json"))):
            err = validate(p)
            if err: bad.append((p, err))
            else: ok += 1
        stray += sorted(glob.glob(os.path.join(d, "*.partial")))
    for p, e in bad: print(f"  BAD    {p}: {e}")
    for p in stray: print(f"  STRAY  {p}: leftover from an interrupted write, safe to delete")
    cfg = []
    for _p in sorted(glob.glob("results/*.json")):
        try: cfg += config_violations(_p, json.load(open(_p)))
        except Exception as _e: cfg.append(f"{_p}: unreadable, {_e}")
    for _c in cfg: print(f"  CONFIG {_c}")
    miss = missing()
    gate = nosmooth_gate()
    if gate: print(f"  GATE   iid_fdsNOSMOOTH_s0: {gate}")
    else: print("  GATE   iid_fdsNOSMOOTH_s0 is bit-identical to same-seed ERM")
    print(f"\n{ok} valid, {len(bad)} bad, {len(stray)} stray partial files"
          + (f"; MANIFEST INCOMPLETE, missing {len(miss)} of {len(EXPECTED)}: "
             + ", ".join(miss[:4]) if miss
             else f"; manifest complete, {len(EXPECTED)} of {len(EXPECTED)}"))
    sys.exit(1 if (bad or stray or miss or gate or cfg) else 0)


if __name__ == "__main__":
    main()
