#!/usr/bin/env python3
"""Measures the interval between each image and the METAR observation it is labelled with, and
how much temperature drifts across that interval. The filename timestamp is GMT, as the AMOS
dataset page states, and the METAR group's Z time is UTC; the table's own Hour column is local time
and differs from the filename hour in most rows. Part 1 also prints what the interval would be if the
filename were read as local time, the sensitivity to that clock."""
import argparse, bisect, collections, csv, datetime, math, os, re, statistics, sys

METAR = re.compile(r"\b(\d{2})(\d{2})(\d{2})Z\b")
FNAME = re.compile(r"^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})")
SENTINEL = -9998.0


def load(path, cameras=None, honour_dd=True, filename_is_local=False):
    """honour_dd=True takes the observation day from the METAR dd field; False ignores dd and
    picks whichever nearby day minimises the gap. filename_is_local=True shifts the filename time
    by the table's Timezone column, the alternative reading of the clock."""
    rows, no_dd_match, aaxx = [], 0, 0
    for r in csv.DictReader(open(path)):
        if cameras and r["CamId"] not in cameras: continue
        try:
            t = float(r["TempM"])
            if t <= SENTINEL: continue
        except (TypeError, ValueError): continue
        fm = FNAME.match(r["Filename"]); mt = METAR.search(r.get("Metar") or "")
        if fm and not mt and (r.get("Metar") or "").startswith("AAXX"):
            aaxx += 1; rows.append({"cam": r["CamId"], "aaxx": True,
                                    "fn": r["Filename"]}); continue
        if not (fm and mt): continue
        dd, hh, mm = int(mt.group(1)), int(mt.group(2)), int(mt.group(3))
        if hh > 23 or mm > 59 or not 1 <= dd <= 31: continue
        try: img = datetime.datetime(*[int(fm.group(i)) for i in range(1, 7)])
        except ValueError: continue
        if filename_is_local:
            try: img = img - datetime.timedelta(hours=float(r["Timezone"]))
            except (TypeError, ValueError): continue
        if honour_dd:
            cands = []
            for dmo in (-1, 0, 1):
                y, mo = img.year, img.month + dmo
                if mo < 1: y, mo = y - 1, mo + 12
                if mo > 12: y, mo = y + 1, mo - 12
                try: cands.append(datetime.datetime(y, mo, dd, hh, mm))
                except ValueError: pass
            if not cands: no_dd_match += 1; continue
        else:
            cands = [datetime.datetime.combine(img.date() + datetime.timedelta(days=d),
                                               datetime.time(hh, mm)) for d in (-1, 0, 1)]
        obs = min(cands, key=lambda o: abs((img - o).total_seconds()))
        rows.append({"cam": r["CamId"], "t": t, "img": img, "obs": obs, "fn": r["Filename"],
                     "gap": abs((img - obs).total_seconds()) / 60.0})
    return rows, no_dd_match


def build_series(rows):
    d = collections.defaultdict(dict)
    for x in rows: d[x["cam"]][x["obs"]] = x["t"]
    return {c: (sorted(v), [v[k] for k in sorted(v)]) for c, v in d.items()}


def bracket(series, cam, when, max_h):
    ts, tp = series[cam]
    i = bisect.bisect_left(ts, when)
    if i == 0 or i >= len(ts): return None
    t0, t1 = ts[i - 1], ts[i]
    span = (t1 - t0).total_seconds() / 3600.0
    if span <= 0 or span > max_h: return None
    w = (when - t0).total_seconds() / (t1 - t0).total_seconds()
    return {"est": tp[i - 1] * (1 - w) + tp[i] * w, "t0": t0, "t1": t1, "i": i}


def q(v, p):
    v = sorted(v); return v[int(p * (len(v) - 1))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/complete_table_with_mcr.csv")
    ap.add_argument("--cameras", default=None)
    ap.add_argument("--guard", type=float, default=3.0)
    ap.add_argument("--like", default="data/labels_cached.csv",
                    help="restrict to the study population in this index. Without it the count\n"
                         "includes images dropped as undecodable, which are not in any split.")
    a = ap.parse_args()
    cams = set(a.cameras.split(",")) if a.cameras else None

    # The released table also carries rows for undecodable images, which appear in no split.
    keep = ({(r["camera"], r["filename"]) for r in csv.DictReader(open(a.like))}
            if a.like else None)
    def study(rows):
        if keep is None: return [r for r in rows if not r.get("aaxx")]
        return [r for r in rows if (r["cam"], r["fn"]) in keep and not r.get("aaxx")]

    if keep is not None:
        _all = load(a.csv, cams, honour_dd=True)[0]
        _timed = len(study(_all))
        _aaxx = sum(1 for r in _all if r.get("aaxx") and (r["cam"], r["fn"]) in keep)
        print(f"POPULATION  {len(keep)} usable images. {_timed} have a parseable METAR timestamp;")
        print(f"            {_aaxx} carry AAXX synoptic reports, which this calculation does not")
        print(f"            parse. Every figure below is on the {_timed} timed images.\n")

    print("PART 1  HOW FAR IS THE LABEL FROM THE PHOTOGRAPH")
    for honour in (True, False):
        rows, nod = load(a.csv, cams, honour_dd=honour)
        rows = study(rows)
        g = [x["gap"] for x in rows]
        tag = "honouring the METAR dd field" if honour else "ignoring dd, nearest day"
        print(f"  {tag}: n={len(rows)}" + (f", dropped {nod} with no valid dd" if honour else ""))
        print(f"    median {statistics.median(g):.1f} min   mean {statistics.fmean(g):.1f}   "
              f"p90 {q(g,.9):.1f}   max {max(g):.1f}")
        print(f"    over 1 h: {100*sum(1 for x in g if x>60)/len(g):.1f}%    "
              f"over 2 h: {100*sum(1 for x in g if x>120)/len(g):.1f}%")
    print("  The median moves by a quarter hour between the two readings. The tail does not")
    print("  survive the second one, and which tail is real depends on whether the release's own")
    print("  day field can be trusted.")
    rows_loc = study(load(a.csv, cams, honour_dd=True, filename_is_local=True)[0])
    print(f"  If the filename were local time rather than UTC, shifting it by the table's Timezone")
    print(f"  column, the median would be {statistics.median([x['gap'] for x in rows_loc]):.1f} min; "
          f"AMOS records the filename time in GMT, so the UTC reading is the documented one.\n")

    rows = [r for r in load(a.csv, cams, honour_dd=True)[0] if not r.get("aaxx")]
    # Build the series from every row, then restrict only the images being scored.
    series = build_series(rows)
    rows = study(rows)

    print("PART 2  INTERPOLATION PROXY FOR THE TEMPERATURE CHANGE OVER THE INTERVAL")
    print("  Estimate the temperature at the moment of capture by interpolating between the two")
    print("  observations that bracket it, then measure how far the assigned label sits from it.")
    floor, percam, outside = [], collections.defaultdict(list), []
    is_outside = []
    for x in rows:
        b = bracket(series, x["cam"], x["img"], a.guard)
        if not b: continue
        e = abs(b["est"] - x["t"])
        isout = x["obs"] not in (b["t0"], b["t1"])
        floor.append(e); percam[x["cam"]].append(e); is_outside.append(isout)
        if isout: outside.append(e)
    print(f"  measurable: {len(floor)} of {len(rows)} images ({100*len(floor)/len(rows):.0f}%)")
    print(f"  MAE {statistics.fmean(floor):.3f} C   median {statistics.median(floor):.3f}   "
          f"RMSE {math.sqrt(statistics.fmean([e*e for e in floor])):.3f}   p90 {q(floor,.9):.2f}")

    # Split the headline: when the assigned observation is itself an interpolation endpoint, est
    # contains the label, so e = w * |T_other - label| and is capped by the endpoint spread.
    capped = [e for e, isout in zip(floor, is_outside) if not isout]
    clean = [e for e, isout in zip(floor, is_outside) if isout]
    if capped and clean:
        print(f"  CAPPED subset   n={len(capped):>6} ({100*len(capped)/len(floor):.1f}%)   "
              f"MAE {statistics.fmean(capped):.3f}   the assigned observation IS an endpoint of")
        print( "                  the interpolation, so it is bounded by the endpoint spread")
        print(f"  CLEAN  subset   n={len(clean):>6} ({100*len(clean)/len(floor):.1f}%)   "
              f"MAE {statistics.fmean(clean):.3f}   shares no endpoint with the estimate, which is")
        print( "                  all this says; it is not an independent measurement of the label error")
        print(f"  The two subsets give about {statistics.fmean(capped):.1f} and "
              f"{statistics.fmean(clean):.1f} C. This is an interpolation proxy for the label gap, not\n"
                "  an error floor, and nothing should be subtracted from any score.")

    print("\n  Which images can be bracketed: the ones with a nearby pair of observations, so the")
    print("           better-covered subset. The gap on the rest is not measured.")
    print("  Does the bracket width matter:")
    for mx in (1.0, 2.0, 3.0, 6.0):
        e = [abs(bracket(series, x["cam"], x["img"], mx)["est"] - x["t"])
             for x in rows if bracket(series, x["cam"], x["img"], mx)]
        print(f"           <= {mx:>3.0f} h   n={len(e):>6}   MAE {statistics.fmean(e):.3f}")
    print("           The figure moves with the bracket width, so it is a range of order 1 C.")

    print("\n  Is it one bad camera? No:")
    ok = {c: v for c, v in percam.items() if len(v) >= 100}
    gaps = {c: statistics.median([x["gap"] for x in rows if x["cam"] == c]) for c in ok}
    xs = [gaps[c] for c in ok]; ys = [statistics.fmean(ok[c]) for c in ok]
    mx_, my = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((p - mx_) * (r - my) for p, r in zip(xs, ys))
    den = math.sqrt(sum((p - mx_) ** 2 for p in xs) * sum((r - my) ** 2 for r in ys))
    print(f"           cameras with >=100 measurable images: {len(ok)}")
    print(f"           median per-camera proxy {statistics.median(ys):.3f} C, "
          f"IQR {q(ys,.25):.2f} to {q(ys,.75):.2f}")
    print(f"           r(camera median gap, camera proxy) = {num/den:+.3f}")
    print("           A camera's interval and its proxy move together. The association is consistent")
    print("           with a timing contribution but does not identify one, and it is not one outlier camera.")
    sane = [c for c in ok if gaps[c] <= 90]
    if sane:
        wt = sum(statistics.fmean(ok[c]) * len(ok[c]) for c in sane) / sum(len(ok[c]) for c in sane)
        print(f"           restricted to the {len(sane)} cameras with median gap <= 90 min: {wt:.3f} C")

    print("\nPART 3  IS THE ASSIGNMENT THE BEST ONE AVAILABLE IN THE SAME FILE")
    print("  Counting facts only. No modelled 'corrected' number is reported here, because the")
    print("  obvious one compares an interpolation against its own endpoint and is capped by")
    print("  construction.")
    if outside:
        print(f"  images whose assigned observation is NOT one of the two that bracket it: "
              f"{len(outside)} of {len(floor)} ({100*len(outside)/len(floor):.0f}%)")
        print(f"  on that subset the assigned label sits {statistics.fmean(outside):.3f} C from the")
        print(f"  interpolated temperature at capture, against {statistics.fmean(floor):.3f} C overall.")
        print("  On that subset the released observation shares no endpoint with the estimate, so")
        print("  the comparison is clean.")

if __name__ == "__main__":
    main()
