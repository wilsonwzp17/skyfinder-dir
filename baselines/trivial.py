#!/usr/bin/env python3
"""No-pixel baselines: a camera-keyed lookup table, a camera-free calendar table, the global
training mean, and the camera-only mean used by the decomposition printout."""
import argparse, collections, csv, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import BIN_WIDTH, shot_fn, dsr_table, format_table


def load(path):
    rows = []
    for r in csv.DictReader(open(path)):
        f = r["filename"]
        rows.append({"filename": f, "camera": r["camera"], "split": r["split"],
                     "temperature": float(r["temperature"]),
                     "month": int(f[4:6]), "hour": int(f[9:11])})
    return rows


def _means(train, keyfns):
    tbls = [collections.defaultdict(list) for _ in keyfns]
    g = []
    for r in train:
        for t, kf in zip(tbls, keyfns): t[kf(r)].append(r["temperature"])
        g.append(r["temperature"])
    return [{k: statistics.fmean(v) for k, v in t.items()} for t in tbls], statistics.fmean(g)


def fit_arms(train):
    cam_keys = [lambda r: (r["camera"], r["month"], r["hour"] // 3),
                lambda r: (r["camera"], r["month"]),
                lambda r: r["camera"]]
    cal_keys = [lambda r: (r["month"], r["hour"] // 3),
                lambda r: r["month"]]
    cam_tbls, gm = _means(train, cam_keys)
    cal_tbls, _ = _means(train, cal_keys)
    levels = collections.Counter()

    def make(tbls, keyfns, tag):
        def predict(r):
            for i, (t, kf) in enumerate(zip(tbls, keyfns)):
                v = t.get(kf(r))
                if v is not None:
                    levels[(tag, i)] += 1
                    return v
            levels[(tag, "global")] += 1
            return gm
        return predict
    return make(cam_tbls, cam_keys, "LOOKUP"), make(cal_tbls, cal_keys, "CALENDAR"), gm, levels


def camera_arm(train):
    """Mean of the row's own camera, backing off to the global mean."""
    tbls, gm = _means(train, [lambda r: r["camera"]])
    def predict(r):
        v = tbls[0].get(r["camera"])
        return gm if v is None else v
    return predict


def fold_means(folds=(0, 1, 2, 3)):
    """Print each arm's MAE per camera-held-out fold, plus the unweighted mean and range."""
    import statistics as _st
    rows = {}
    for f in folds:
        r = load(f"data/splits_cd{f}.csv")
        tr = [x for x in r if x["split"] == "train"]
        te = [x for x in r if x["split"] == "test"]
        pl, pc, gm, _ = fit_arms(tr)
        m = lambda fn: _st.fmean([abs(fn(x) - x["temperature"]) for x in te])
        rows[f] = (m(pl), m(pc), _st.fmean([abs(gm - x["temperature"]) for x in te]))
    print("\nCAMERA HELD OUT, per fold and unweighted mean")
    print(f"  {'fold':<6}{'LOOKUP':>9}{'CALENDAR':>10}{'GLOBAL':>9}")
    for f, v in sorted(rows.items()):
        print(f"  {f:<6}{v[0]:>9.2f}{v[1]:>10.2f}{v[2]:>9.2f}")
    cols = list(zip(*rows.values()))
    print(f"  {'mean':<6}" + "".join(f"{_st.fmean(c):>9.2f}" if i != 1 else f"{_st.fmean(c):>10.2f}"
                                    for i, c in enumerate(cols)))
    print(f"  {'range':<6}" + "".join(f"{min(c):>6.2f}-{max(c):<4.2f}" for c in cols))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/splits_iid.csv")
    ap.add_argument("--basis", choices=["group", "bin"], default="bin",
                    help="shot-region basis: bin is the DIR definition and the one the report uses; "
                         "group is the joint (bin, camera) axis and is degenerate under a "
                         "camera-disjoint split")
    ap.add_argument("--fold-means", action="store_true",
                    help="print the per-fold and unweighted-mean control table")
    a = ap.parse_args()
    if a.fold_means: fold_means(); return
    rows = load(a.splits)
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    p_lookup, p_cal, gmean, levels = fit_arms(train)
    p_cam = camera_arm(train)
    err_lookup = [p_lookup(r) - r["temperature"] for r in test]
    err_cal = [p_cal(r) - r["temperature"] for r in test]
    err_global = [gmean - r["temperature"] for r in test]

    region, _ = shot_fn(train, BIN_WIDTH, basis=a.basis)
    disjoint = not ({r["camera"] for r in train} & {r["camera"] for r in test})
    print(f"split {a.splits}   train {len(train)}   test {len(test)}")
    print(f"shot-region basis: {a.basis}"
          + ("  (joint (bin, camera) groups)" if a.basis == "group"
             else "  (TARGET BINS, the DIR definition)")
          + ("; every test group is unseen under a camera-disjoint split, so group regions are "
             "degenerate here" if disjoint else ""))
    print(f"backoff usage: {dict(levels)}   <- 'global' means the table had no entry at all")
    print()
    for errs, name in ((err_lookup, "LOOKUP    camera + month + 3-hour block  (camera-keyed)"),
                       (err_cal,    "CALENDAR  month + 3-hour block           (camera-free)"),
                       (err_global, "GLOBAL    training mean                  (global-mean baseline)")):
        print(format_table(dsr_table(errs, test, region), name)); print()
    m = lambda e: sum(abs(x) for x in e) / len(e)
    lo, ca, gl = m(err_lookup), m(err_cal), m(err_global)
    print(f"  margin over GLOBAL:  LOOKUP {100*(gl-lo)/gl:+.1f}%   CALENDAR {100*(gl-ca)/gl:+.1f}%")

    # Both orderings are printed because the decomposition is order-dependent.
    cm = m([p_cam(r) - r["temperature"] for r in test])
    gap = gl - lo
    if gap > 1e-9:
        print(f"  CAMERA only (backing off to the global mean): {cm:.3f}")
        print(f"  closing the {gap:.3f} C gap from GLOBAL to LOOKUP, two equally available paths:")
        print(f"    calendar first  {gl-ca:6.3f} ({100*(gl-ca)/gap:4.1f}%)  then camera   "
              f"{ca-lo:6.3f} ({100*(ca-lo)/gap:4.1f}%)")
        print(f"    camera first    {gl-cm:6.3f} ({100*(gl-cm)/gap:4.1f}%)  then calendar "
              f"{cm-lo:6.3f} ({100*(cm-lo)/gap:4.1f}%)")
        inter = (gl - ca) - (cm - lo)
        print(f"    interaction {inter:+.3f} C, larger in magnitude than the camera-first main")
        print(f"    effect of {gl-cm:.3f}, so neither path may be reported as two causes.")
    if abs(lo - gl) < 1e-9:
        print("  LOOKUP is numerically IDENTICAL to GLOBAL: every test row fell through to the")
        print("  global mean because its camera never appears in training. The camera shortcut")
        print("  supplies exactly zero transferable information here.")
    print(f"  The bar an image model must clear is CALENDAR at MAE {ca:.3f}, not GLOBAL.")


if __name__ == "__main__":
    main()
