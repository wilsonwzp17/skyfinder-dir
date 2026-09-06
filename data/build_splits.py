#!/usr/bin/env python3
"""Build train/val/test splits in two regimes: "iid", which caps test then val per group
g = (temperature bin, camera), and "camera-disjoint", which holds out whole cameras."""
import argparse, collections, csv, math, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import BIN_WIDTH, bin_index, group_key, shot_fn, SHOTS


def split_iid(rows, bin_width, test_cap, val_cap, seed):
    by = collections.defaultdict(list)
    for r in rows:
        by[group_key(r["temperature"], r["camera"], bin_width)].append(r)
    rnd = random.Random(seed)
    tr, va, te = [], [], []
    for key in sorted(by):
        g = by[key][:]
        rnd.shuffle(g)
        te += g[:test_cap]
        va += g[test_cap:test_cap + val_cap]
        tr += g[test_cap + val_cap:]
    return tr, va, te


def split_camera_disjoint(rows, fold, n_folds, n_val, seed):
    """Partition the cameras into n_folds test blocks, with val cameras drawn from the rest."""
    cams = sorted({r["camera"] for r in rows})
    rnd = random.Random(seed)
    order = cams[:]
    rnd.shuffle(order)
    blocks = [order[i::n_folds] for i in range(n_folds)]
    test_cams = set(blocks[fold % n_folds])
    rest = [c for c in order if c not in test_cams]
    val_cams = set(rest[:n_val])
    tr = [r for r in rows if r["camera"] not in test_cams and r["camera"] not in val_cams]
    va = [r for r in rows if r["camera"] in val_cams]
    te = [r for r in rows if r["camera"] in test_cams]
    return tr, va, te, sorted(test_cams), sorted(val_cams)


def fit_caps(rows, bin_width, target):
    cells = list(collections.Counter(
        group_key(r["temperature"], r["camera"], bin_width) for r in rows).values())
    best = None
    for ct in range(1, 40):
        for cv in range(1, 60):
            te = sum(min(ct, n) for n in cells)
            va = sum(min(cv, max(0, n - ct)) for n in cells)
            tr = sum(cells) - te - va
            e = abs(tr - target[0]) + abs(va - target[1]) + abs(te - target[2])
            if best is None or e < best[0]: best = (e, ct, cv, tr, va, te)
    return best


def describe(tr, va, te, bin_width, label, basis="group"):
    print(f"{label}: train {len(tr)}  val {len(va)}  test {len(te)}")
    if not te: return
    region, counts = shot_fn(tr, bin_width, basis=basis)
    comp = collections.Counter(region(r) for r in te)
    n = len(te)
    tag = "groups per Eq.(1)" if basis == "group" else "TARGET BINS, not groups (see below)"
    print(f"  test by shot region, {tag}: "
          + ", ".join(f"{s} {comp[s]} ({100*comp[s]/n:.1f}%)" for s in SHOTS))
    if basis == "bin":
        print("  NOTE: under a camera-disjoint split every test group is unseen in training by")
        print("        construction, so the group-level breakdown is 100 percent zero-shot and")
        print("        carries no information. Regions here are over target bins. The two bases")
        print("        are NOT comparable across regimes.")
    tr_groups = {group_key(r["temperature"], r["camera"], bin_width) for r in tr}
    te_groups = {group_key(r["temperature"], r["camera"], bin_width) for r in te}
    print(f"  groups: {len(tr_groups)} in train, {len(te_groups)} in test, "
          f"{len(te_groups - tr_groups)} test groups unseen in train")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True,
                    help="the post-cache index, data/labels_cached.csv. There is no default: an\n"
                         "all-camera label file would build a split the stored results cannot match.")
    ap.add_argument("--out", default="data/splits.csv")
    ap.add_argument("--regime", choices=["iid", "camera-disjoint"], default="iid")
    ap.add_argument("--bin-width", type=float, default=BIN_WIDTH)
    ap.add_argument("--test-cap", type=int, default=3)
    ap.add_argument("--val-cap", type=int, default=4)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=4)
    ap.add_argument("--n-val-cameras", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cameras", default=None)
    ap.add_argument("--fit-to", default=None,
                    help="train,val,test sizes to fit the caps against; used once to choose the "
                         "default caps of 3 and 4 by fitting the benchmark paper's split shares")
    a = ap.parse_args()

    rows = [{"filename": r["filename"], "camera": r["camera"],
             "temperature": float(r["temperature"])}
            for r in csv.DictReader(open(a.labels))]
    if a.cameras:
        keep = set(a.cameras.split(","))
        rows = [r for r in rows if r["camera"] in keep]
    print(f"population: {len(rows)} images, {len({r['camera'] for r in rows})} cameras")

    if a.regime == "iid":
        if a.fit_to:
            t = tuple(int(x) for x in a.fit_to.split(","))
            e, ct, cv, *_ = fit_caps(rows, a.bin_width, t)
            print(f"caps fitted to {t}: test {ct} / val {cv}")
            a.test_cap, a.val_cap = ct, cv
        tr, va, te = split_iid(rows, a.bin_width, a.test_cap, a.val_cap, a.seed)
        label = f"iid, cap {a.test_cap}/{a.val_cap} per group, seed {a.seed}"
        meta = ""
    else:
        tr, va, te, tc, vc = split_camera_disjoint(rows, a.fold, a.n_folds,
                                                   a.n_val_cameras, a.seed)
        label = f"camera-disjoint, fold {a.fold} of {a.n_folds}, seed {a.seed}"
        meta = f"  test cameras {tc}\n  val cameras  {vc}"
        print(meta)

    describe(tr, va, te, a.bin_width, label,
             basis="bin" if a.regime == "camera-disjoint" else "group")

    # the key is (camera, filename): basenames repeat across cameras
    keys = [(r["camera"], r["filename"]) for g in (tr, va, te) for r in g]
    assert len(keys) == len(set(keys)), "duplicate (camera, filename) across splits"
    if a.regime == "camera-disjoint":
        assert not ({r["camera"] for r in tr} & {r["camera"] for r in te}), "camera leak"

    if a.out != "/dev/null":
        with open(a.out, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["filename", "camera", "temperature", "split"])
            for name, grp in (("train", tr), ("val", va), ("test", te)):
                for r in grp:
                    w.writerow([r["filename"], r["camera"], f"{r['temperature']:.1f}", name])
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
