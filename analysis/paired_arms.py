#!/usr/bin/env python3
"""Compare two arms paired per image, overall and by shot region."""
import argparse, csv, glob, json, math, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import BIN_WIDTH, shot_fn, SHOTS


def arm_errors(results_dir, prefix, arm):
    rs = [json.load(open(f)) for f in sorted(glob.glob(f"{results_dir}/{prefix}_{arm}_s*.json"))]
    if not rs: return None, []
    n = {len(r["errors"]) for r in rs}
    if len(n) != 1:
        sys.exit(f"{arm}: runs disagree on test size {n}; they are not comparable")
    per_image = [statistics.fmean(c) for c in zip(*[[abs(e) for e in r["errors"]] for r in rs])]
    return per_image, rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="erm")
    ap.add_argument("--b", default="lds")
    ap.add_argument("--prefix", default="iid")
    ap.add_argument("--results", default="results")
    ap.add_argument("--splits", default="data/splits_iid.csv")
    a = ap.parse_args()

    ea, ra = arm_errors(a.results, a.prefix, a.a)
    eb, rb = arm_errors(a.results, a.prefix, a.b)
    if ea is None or eb is None: sys.exit(f"missing results for {a.a} or {a.b}")
    if len(ea) != len(eb): sys.exit("arms evaluated on different test sets; not comparable")

    shas = {r.get("split_sha256") for r in ra + rb if r.get("split_sha256")}
    if len(shas) > 1: sys.exit("arms were run against different split CONTENT; not comparable")

    rows = list(csv.DictReader(open(a.splits)))
    tr = [{"camera": r["camera"], "temperature": float(r["temperature"])}
          for r in rows if r["split"] == "train"]
    te = [{"camera": r["camera"], "temperature": float(r["temperature"])}
          for r in rows if r["split"] == "test"]
    # basis defaults to "bin", so regions come from target bins rather than (bin, camera) groups.
    region, _ = shot_fn(tr, BIN_WIDTH)
    regs = [region(r) for r in te]

    print(f"{a.b} minus {a.a}, paired per image. Negative means {a.b} is better.")
    print(f"seeds: {a.a} {len(ra)}, {a.b} {len(rb)}   test images {len(ea)}")
    print(f"per-seed Overall  {a.a}: {[round(r['table']['Overall'][0],3) for r in ra]}")
    print(f"per-seed Overall  {a.b}: {[round(r['table']['Overall'][0],3) for r in rb]}")
    # The images are not independent: they come from 12 cameras and a camera repeats. A per-image
    # standard error would understate the spread, so only the camera-clustered errors are printed.
    print(f"\n  {'region':<9}{'n':>6}{'mean diff':>12}   lower")
    diffs = {}
    for name, sel in [("ALL", [True] * len(te))] + [(s, [r == s for r in regs]) for s in SHOTS]:
        d = [y - x for x, y, k in zip(ea, eb, sel) if k]
        if len(d) < 2:
            print(f"  {name:<9}{len(d):>6}   too few"); continue
        m = statistics.fmean(d)
        diffs[name] = m
        print(f"  {name:<9}{len(d):>6}{m:>+12.3f}   {a.b if m < 0 else a.a}")

    # Camera-clustered standard errors; CR1 adds the finite-cluster correction sqrt(G/(G-1)).
    cams = [r["camera"] for r in te]
    print(f"\n  {'region':<9}{'clusters':>9}{'mean diff':>12}{'CR0':>9}{'CR1':>9}")
    for name, sel in [(s, [r == s for r in regs]) for s in SHOTS]:
        idx = [i for i, k in enumerate(sel) if k]
        if len(idx) < 2: continue
        d = [eb[i] - ea[i] for i in idx]
        m = statistics.fmean(d)
        by = {}
        for i in idx: by.setdefault(cams[i], []).append(eb[i] - ea[i])
        G = len(by)
        if G < 2: continue
        ss = sum((sum(v) - len(v) * m) ** 2 for v in by.values())
        cr0 = math.sqrt(ss) / len(d)
        cr1 = cr0 * math.sqrt(G / (G - 1))
        print(f"  {name:<9}{G:>9}{m:>+12.3f}{cr0:>9.3f}{cr1:>9.3f}")

    # Recount zero-shot on the other axis, where groups are (bin, camera) rather than bins.
    from common import shot_fn as _sf
    _rg, _ = _sf(tr, BIN_WIDTH, basis="group")
    _gz = [r for r in te if _rg(r) == "zero"]
    _bz = sum(1 for r in _gz if region(r) == "zero")
    print(f"\n  joint (bin, camera) axis: {len(_gz)} of {len(te)} test images are zero-shot; "
          f"{_bz} of those")
    print(f"  sit in a target bin with no training data at all.")

    print("\n  Five regions are compared with no multiplicity correction. The report makes no claim")
    print("  from the ordering across regions; what it rests on is the per-seed sign agreement below.")

    # Per-seed differences, and how many seeds agree with the sign of the mean.
    bysd = lambda rs: {r["seed"]: [abs(e) for e in r["errors"]] for r in rs}
    Ea, Eb = bysd(ra), bysd(rb)
    seeds = sorted(set(Ea) & set(Eb))
    print(f"\n  per-seed paired difference, and how many seeds favour {a.b}:")
    print(f"  {'region':<9}" + "".join(f"{'seed '+str(s):>10}" for s in seeds) + f"{'better':>9}")
    for name in SHOTS:
        sel = [r == name for r in regs]
        idx = [i for i, k in enumerate(sel) if k]
        if not idx: continue
        per = [statistics.fmean(Eb[s][i] - Ea[s][i] for i in idx) for s in seeds]
        m = statistics.fmean(per)
        better = sum(1 for v in per if v < 0)
        print(f"  {name:<9}" + "".join(f"{v:>+10.3f}" for v in per)
              + f"{better:>6} of {len(per)}")


if __name__ == "__main__":
    main()
