#!/usr/bin/env python3
"""Generate the ablation tables from results/ and inject them into the report.

Markers: <!-- BEGIN:name --> ... <!-- END:name -->
"""
import collections, glob, json, os, re, sys

REPORT = "report/REPORT.md"
ARM_ORDER = ["erm", "sqinv", "lds", "fds", "lds_fds"]
ARM_NAME = {"erm": "ERM", "sqinv": "SQINV", "lds": "SQINV + LDS", "fds": "FDS",
            "lds_fds": "SQINV + LDS + FDS"}


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import is_diagnostic, gm



def regional(basis):
    """Recompute the ablation cells on the given shot basis from the stored per-image errors."""
    import csv as _c, statistics as _s
    from common import BIN_WIDTH, shot_fn
    sp = list(_c.DictReader(open("data/splits_iid.csv")))
    tr = [{"camera": r["camera"], "temperature": float(r["temperature"])} for r in sp if r["split"] == "train"]
    te = [{"camera": r["camera"], "temperature": float(r["temperature"])} for r in sp if r["split"] == "test"]
    reg, _ = shot_fn(tr, BIN_WIDTH, basis=basis)
    idx = {}
    for i, r in enumerate(te): idx.setdefault(reg(r), []).append(i)
    out = {}
    for arm in ARM_ORDER:
        rs = [json.load(open(f)) for f in sorted(glob.glob(f"results/iid_{arm}_s*.json"))]
        rs = [r for r in rs if not is_diagnostic(r)]
        if not rs: continue
        E = [[abs(e) for e in r["errors"]] for r in rs]
        cells = {}
        # each cell: mean MAE across seeds, its standard deviation, the cell size, and the mean
        # across seeds of the error geometric mean of the same runs
        for k in ("many", "medium", "few", "zero"):
            if k not in idx: continue
            per = [_s.fmean(e[i] for i in idx[k]) for e in E]
            g = [gm([e[i] for i in idx[k]]) for e in E]
            cells[k] = (_s.fmean(per), _s.stdev(per) if len(per) > 1 else 0.0, len(idx[k]), _s.fmean(g))
        ov = [_s.fmean(e) for e in E]
        cells["overall"] = (_s.fmean(ov), _s.stdev(ov) if len(ov) > 1 else 0.0, len(te),
                            _s.fmean(gm(e) for e in E))
        out[arm] = (cells, E)
    signs = {}
    if "erm" in out:
        Eerm = out["erm"][1]
        for arm in out:
            if arm == "erm": continue
            signs[arm] = {}
            for k in ("many", "medium", "few", "zero"):
                if k not in idx: continue
                if len(out[arm][1]) != len(Eerm):
                    raise SystemExit(
                        f"{arm} has {len(out[arm][1])} seeds against ERM's {len(Eerm)}; refusing "
                        "to write a table that compares unequal seed sets")
                d = [_s.fmean(out[arm][1][j][i] - Eerm[j][i] for i in idx[k])
                     for j in range(len(Eerm))]
                signs[arm][k] = (_s.fmean(d), sum(1 for x in d if x < 0), len(d))
    return out, signs, {k: len(v) for k, v in idx.items()}


def shot_table(basis, uniform_seeds):
    """The in-distribution table. The caption lives in the report; this writes the rows only.
    An arm whose seed count differs from the rest carries that count, so a mid-sweep snapshot
    cannot be read as a finished result."""
    out, signs, ns = regional(basis)
    if not out: return None, None
    cols = ["overall", "many", "medium", "few", "zero"]
    head = "| Arm | " + " | ".join(
        f"{c.capitalize()}, n={ns.get(c, 0) if c != 'overall' else sum(ns.values()):,}"
        for c in cols) + " |"
    body = [head, "|" + "---|" * (len(cols) + 1)]
    for arm in ARM_ORDER:
        if arm not in out: continue
        cells = out[arm][0]
        row = [ARM_NAME[arm] + ("" if uniform_seeds else f" ({len(out[arm][1])} seeds)")]
        for c in cols:
            if c not in cells: row.append("n/a"); continue
            m, sd, _n, g = cells[c]
            row.append((f"{m:.2f} ±{sd:.2f}" if sd else f"{m:.2f}") + f" / {g:.2f}")
        body.append("| " + " | ".join(row) + " |")
    return "\n".join(body), signs


def load(d):
    g = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(f))
        if is_diagnostic(r): continue
        base = os.path.basename(r["splits"]).replace("splits_", "").replace(".csv", "")
        g[(base, r["arm"])].append(r)
    return g


def inject(text, name, body):
    pat = re.compile(rf"(<!-- BEGIN:{name} -->\n)(?:.*?\n)??(<!-- END:{name} -->)", re.S)
    if not pat.search(text):
        print(f"  marker {name} not found, skipped"); return text
    return pat.sub(lambda m: m.group(1) + body + "\n" + m.group(2), text)


def main():
    rep = open(REPORT).read()
    g = load("results")

    iid = sorted([(a, v) for (b, a), v in g.items() if b == "iid"],
                 key=lambda kv: ARM_ORDER.index(kv[0]))
    if iid:
        counts = {a: len(v) for a, v in iid}
        uniform = len(set(counts.values())) == 1
        if min(counts.values()) < 3:
            print(f"  WARNING iid_ablation: an arm has fewer than 3 seeds {counts}. The table "
                  f"will say so, but this is a mid-sweep snapshot, not a final result.")
        tb, signs = shot_table("bin", uniform)
        if tb: rep = inject(rep, "iid_ablation", tb)
        print(f"  iid_ablation: {counts}" + ("" if uniform else "   <- NOT YET UNIFORM"))

    # Camera-disjoint folds are listed individually; the spread between them is larger than the
    # gaps between methods, so a pooled mean would hide the result.
    import csv as _csv, statistics as _st
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from baselines.trivial import load as _bl, fit_arms as _fa
    folds = sorted(int(re.search(r"cd(\d+)_", os.path.basename(f)).group(1))
                   for f in glob.glob("results/cd*_erm_s0.json"))
    if folds:
        body = ["| Fold | Test images | CALENDAR | GLOBAL | ERM | SQINV | SQINV + LDS | SQINV + LDS + FDS |",
                "|---|---|---|---|---|---|---|---|"]
        for fd in sorted(set(folds)):
            sp = f"data/splits_cd{fd}.csv"
            rws = _bl(sp); tr = [x for x in rws if x["split"] == "train"]
            te = [x for x in rws if x["split"] == "test"]
            _, pc, gm, _ = _fa(tr)
            cal = _st.fmean(abs(pc(x) - x["temperature"]) for x in te)
            glo = _st.fmean(abs(gm - x["temperature"]) for x in te)
            def g(a):
                q = f"results/cd{fd}_{a}_s0.json"
                return f"{json.load(open(q))['table']['Overall'][0]:.2f}" if os.path.exists(q) else "pending"
            body.append(f"| {fd} | {len(te):,} | {cal:.2f} | {glo:.2f} | {g('erm')} | {g('sqinv')} | {g('lds')} | {g('lds_fds')} |")
        rep = inject(rep, "cd_ablation", "\n".join(body))
        print(f"  cd_ablation: {len(set(folds))} folds, reported individually")

    open(REPORT, "w").write(rep)
    print(f"\n{REPORT}: {len(rep.split())} words, {rep.count('pending')} pending cells remaining")


if __name__ == "__main__":
    main()
