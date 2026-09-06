#!/usr/bin/env python3
"""Leakage probes for a split. P0 and P1 are structural and must hold; P2 to P4 are
measurements printed without a threshold."""
import argparse, bisect, collections, csv, datetime, statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/splits_iid.csv")
    ap.add_argument("--control-mae", type=float, default=None)
    a = ap.parse_args()
    rows = [{"f": r["filename"], "c": r["camera"], "t": float(r["temperature"]),
             "s": r["split"], "d": datetime.datetime.strptime(r["filename"][:15], "%Y%m%d_%H%M%S")}
            for r in csv.DictReader(open(a.splits))]
    tr = [r for r in rows if r["s"] == "train"]
    te = [r for r in rows if r["s"] == "test"]
    va = [r for r in rows if r["s"] == "val"]
    disjoint = not ({r["c"] for r in tr} & {r["c"] for r in te})
    print(f"{a.splits}: train {len(tr)}  val {len(va)}  test {len(te)}"
          f"{'   (camera-disjoint)' if disjoint else ''}\n")
    V = []

    byname = collections.defaultdict(set)
    for r in rows: byname[r["f"]].add(r["c"])
    collide = sum(1 for v in byname.values() if len(v) > 1)
    keys = [(r["c"], r["f"]) for r in rows]
    dup = len(keys) - len(set(keys))
    V.append(("P0 KEY INTEGRITY", dup == 0,
              f"{dup} duplicate (camera, filename) rows; {collide} basenames repeat across "
              f"cameras, which is why the key has two parts"))

    ks = [{(r["c"], r["f"]) for r in g} for g in (tr, va, te)]
    ov = len(ks[0] & ks[2]) + len(ks[0] & ks[1]) + len(ks[1] & ks[2])
    V.append(("P1 DISJOINTNESS", ov == 0, f"{ov} shared (camera, filename) keys"))

    if disjoint:
        for n in ("P2 TEMPORAL GAP", "P3 NEIGHBOUR ORACLE", "P4 SAME-DAY"):
            V.append((n, None, "vacuous: no test camera appears in training"))
    else:
        idx = collections.defaultdict(list)
        for r in tr: idx[r["c"]].append((r["d"], r["t"]))
        for c in idx: idx[c].sort()
        gaps, oerr = [], []
        for r in te:
            arr = idx.get(r["c"])
            if not arr: continue
            ds = [x[0] for x in arr]
            i = bisect.bisect_left(ds, r["d"])
            cand = [j for j in (i - 1, i) if 0 <= j < len(arr)]
            if not cand: continue
            j = min(cand, key=lambda j: abs((arr[j][0] - r["d"]).total_seconds()))
            gaps.append(abs((arr[j][0] - r["d"]).total_seconds()) / 60.0)
            oerr.append(abs(arr[j][1] - r["t"]))
        med = statistics.median(gaps)
        V.append(("P2 TEMPORAL GAP", None,
                  f"median {med:.1f} min to the nearest training frame; "
                  f"under 10 min: {100*sum(1 for g in gaps if g<10)/len(gaps):.1f}%"))
        # P3b re-runs P3 using only the most recent earlier frame, never a later one.
        perr, fallback = [], 0
        for r in te:
            arr = idx.get(r["c"])
            if not arr: continue
            ds = [x[0] for x in arr]
            i = bisect.bisect_left(ds, r["d"])
            if i - 1 >= 0:
                perr.append(abs(arr[i - 1][1] - r["t"]))
            elif i < len(arr):
                perr.append(abs(arr[i][1] - r["t"])); fallback += 1
        om = statistics.fmean(oerr)
        if a.control_mae:
            V.append(("P3 NEIGHBOUR ORACLE", None,
                      f"oracle MAE {om:.3f} vs control {a.control_mae:.3f} "
                      f"({100*(a.control_mae-om)/a.control_mae:+.1f}% better)"))
            V.append(("P3b PAST-ONLY ORACLE", None,
                      f"past-only MAE {statistics.fmean(perr):.3f} over {len(perr)} images; "
                      f"{fallback} had no earlier frame and used a later one"))
        else:
            V.append(("P3 NEIGHBOUR ORACLE", None, f"oracle MAE {om:.3f} (no control supplied)"))
        days = collections.defaultdict(set)
        for r in tr: days[r["c"]].add(r["d"].date())
        frac = sum(1 for r in te if r["d"].date() in days.get(r["c"], ())) / len(te)
        V.append(("P4 SAME-DAY", None,
                  f"{100*frac:.1f}% share a (camera, day) with a training image"))

    for n, ok, d in V:
        mark = "ok" if ok else "LEAK" if ok is False else "    "
        print(f"  [{mark:<4}] {n:<22} {d}")
    nf = sum(1 for _, ok, _ in V if ok is False)
    print(f"\n  P0 and P1 are structural and either hold or do not. P2, P3 and P4 are measurements\n"
          f"  reported without a threshold: what counts as too much leakage depends on the claim\n"
          f"  being made, and the report argues from the sizes rather than from a verdict.")


if __name__ == "__main__":
    main()
