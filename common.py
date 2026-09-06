#!/usr/bin/env python3
"""Shared definitions: temperature binning, shot regions, tables and metrics."""
import collections, math

# Binning domain, padded to whole degrees around the released table's range of -27.2 to 50.0 C.
# The twelve study cameras span -13.2 to 39.4 C, so 27 of the 81 bins hold no image from these cameras and the
# FDS bucket range is derived from the training labels rather than from these limits.
TMIN, TMAX = -30, 50
BIN_WIDTH = 1.0                                     # one-degree bins, the reference's one-year granularity carried over
SHIFT = -TMIN
NBINS = int((TMAX - TMIN) / BIN_WIDTH) + 1          # 81 one-degree bins

# The DIR reference shot_metrics thresholds. The zero region, a cell with no training image, is
# the benchmark paper's addition, applied here over marginal bins as well as joint groups.
MANY_THR, FEW_THR = 100, 20


def bin_index(t, bin_width=BIN_WIDTH):
    """Target bin index, clamped to [0, NBINS-1]; floor() and the shift keep negative temperatures valid."""
    return min(NBINS - 1, max(0, int(math.floor(float(t) / bin_width)) + int(SHIFT / bin_width)))


def group_key(t, attr, bin_width=BIN_WIDTH):
    """Group key g = (bin index, attribute)."""
    return (bin_index(t, bin_width), attr)


def shot_fn(train_rows, bin_width=BIN_WIDTH, many=MANY_THR, few=FEW_THR,
            tkey=lambda r: r["temperature"], akey=lambda r: r["camera"], basis="bin"):
    """Return (region_of_row, counts) with counts from train; basis="bin" scores over marginal
    target bins, basis="group" over joint (bin, attribute) groups."""
    assert basis in ("group", "bin")
    keyf = ((lambda r: group_key(tkey(r), akey(r), bin_width)) if basis == "group"
            else (lambda r: (bin_index(tkey(r), bin_width),)))
    counts = collections.Counter(keyf(r) for r in train_rows)
    def region(row):
        n = counts[keyf(row)]
        if n == 0:     return "zero"
        if n > many:   return "many"
        if n >= few:   return "medium"
        return "few"
    return region, counts


def is_diagnostic(r):
    """True for the no-smoothing control run, which is stored with arm="fds" but never smooths."""
    return r.get("start_smooth") is not None and r["start_smooth"] > r.get("epochs", 0)


def mae(errors):
    return sum(abs(e) for e in errors) / len(errors) if errors else float("nan")


def gm(errors):
    """Geometric mean of the absolute errors, the DIR definition: the reference applies
    scipy.stats.gmean to the L1 errors as they are, so a single exact-zero error makes it zero."""
    if not errors: return float("nan")
    if any(e == 0 for e in errors): return 0.0
    return math.exp(sum(math.log(abs(e)) for e in errors) / len(errors))


SHOTS = ("many", "medium", "few", "zero")


def dsr_table(errors, rows, region, akey=lambda r: r["camera"]):
    """Build {column: (mae, gm, n)} following the column order of the benchmark paper, arXiv
    2606.01723: Overall, per-attribute Average and Worst, then each shot region as one cell rather
    than the paper's Average and Worst per region."""
    out = {"Overall": (mae(errors), gm(errors), len(errors))}
    by_attr = collections.defaultdict(list)
    for e, r in zip(errors, rows):
        by_attr[akey(r)].append(e)
    per = {a: mae(v) for a, v in by_attr.items()}
    if per:
        worst = max(per, key=per.get)
        out["Attr-Average"] = (sum(per.values()) / len(per), None, len(per))
        out["Attr-Worst"] = (per[worst], None, len(by_attr[worst]))
        out["_worst_attr"] = worst
    for s in SHOTS:
        sel = [(e, r) for e, r in zip(errors, rows) if region(r) == s]
        out[s.capitalize()] = ((mae([e for e, _ in sel]), gm([e for e, _ in sel]), len(sel))
                               if sel else None)
    return out


def format_table(tbl, title=""):
    lines = []
    if title: lines.append(title)
    lines.append(f"  {'column':<14}{'n':>7}{'MAE':>9}{'GM':>9}")
    for k in ("Overall", "Attr-Average", "Attr-Worst") + tuple(s.capitalize() for s in SHOTS):
        v = tbl.get(k)
        if v is None:
            lines.append(f"  {k:<14}{0:>7}{'n/a':>9}{'n/a':>9}")
            continue
        m, g, n = v
        lines.append(f"  {k:<14}{n:>7}{m:>9.3f}" + (f"{g:>9.3f}" if g is not None else f"{'':>9}"))
    if "_worst_attr" in tbl: lines.append(f"  (worst attribute: {tbl['_worst_attr']})")
    return "\n".join(lines)


class frozen_bn:
    """Keep BatchNorm from updating its running statistics while the model stays in train() mode,
    which FDS needs so that forward returns features."""

    def __init__(self, model):
        self.model = model
        self.frozen = []

    def __enter__(self):
        import torch.nn as nn          # lazy so common stays importable without torch
        self.frozen = [m for m in self.model.modules()
                       if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.training]
        for m in self.frozen: m.eval()
        return self.model

    def __exit__(self, *exc):
        for m in self.frozen: m.train()
        return False
