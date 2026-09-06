#!/usr/bin/env python3
"""Runs the three things that break when the DIR code is given a signed target, then compares how
concentrated the inverse-frequency weights are with and without the reference's count clip."""
import csv, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
from common import NBINS, SHIFT, bin_index


def bug1_keyerror():
    print("BREAK 1  _prepare_weights raises on a negative label.")
    print("  Upstream imdb-wiki-dir/datasets.py builds `{x: 0 for x in range(max_target)}`,")
    print("  whose keys start at 0, then indexes it with `min(max_target - 1, int(label))`.")
    def upstream(labels, max_target=121):
        value_dict = {x: 0 for x in range(max_target)}
        for label in labels:
            value_dict[min(max_target - 1, int(label))] += 1
        return value_dict
    print(f"  ages      [0, 25.5, 60, 100] -> ", end="")
    try: upstream([0, 25.5, 60, 100]); print("OK")
    except KeyError as e: print(f"KeyError {e}")
    print(f"  temps [-13.2, 0, 15, 39.4]  -> ", end="")
    try: upstream([-13.2, 0.0, 15.0, 39.4]); print("OK")
    except KeyError as e: print(f"KeyError {e}   <- reproduced")


def bug2_truncation():
    print("\nBREAK 2  A constant bin shift fixes the crash and leaves a silent error behind.")
    print("  Python's int() truncates TOWARD ZERO, so it is not floor() on negative input.")
    print("  The bin holding zero ends up twice as wide, and every negative bin is displaced.")
    print(f"  {'T':>8}{'int()+shift':>13}{'floor()+shift':>15}")
    diff = 0
    for t in (-13.2, -0.6, -0.4, 0.0, 0.4, 12.7):
        a = min(NBINS - 1, max(0, int(t) + SHIFT)); b = bin_index(t)
        diff += a != b
        print(f"  {t:>8.1f}{a:>13}{b:>15}" + ("   DIFFERS" if a != b else ""))
    print(f"  {diff} of 6 probes disagree. Every disagreement is on the negative half, which is")
    print("  exactly the half where the sparse bins live and therefore what LDS smooths.")


def bug3_fds_labels():
    print("\nBREAK 3  FDS must be fed bin INDICES, not temperatures. Not an upstream defect: an")
    print("  error you make if you pass the obvious thing.")
    import torch
    from vendor.fds import FDS
    path = "data/labels_cached.csv"
    if not os.path.exists(path):
        print("  data/labels_cached.csv not present, skipping"); return
    temps = [float(r["temperature"]) for r in csv.DictReader(open(path))]
    feats = torch.randn(len(temps), 8)
    b0_raw = None
    for tag, labels in (("raw temperatures", torch.tensor(temps, dtype=torch.float32)),
                        ("bin indices    ", torch.tensor([bin_index(t) for t in temps]))):
        f = FDS(feature_dim=8, bucket_num=NBINS, bucket_start=0, start_update=0, start_smooth=1)
        f.update_running_stats(feats, labels, 0)
        occupied = int((f.num_samples_tracked > 0).sum().item())
        tracked = int(f.num_samples_tracked.sum().item())
        b0 = int(f.num_samples_tracked[0].item())
        if tag.startswith("raw"): b0_raw = b0
        print(f"  {tag}: unique labels {len(torch.unique(labels)):>4}   buckets touched {occupied:>3}"
              f"   tracked {tracked:>6}   bucket0 {b0:>5}")
    n_neg = sum(1 for t in temps if t < 0)
    print(f"  With raw temperatures every negative value falls into bucket 0, because the")
    print(f"  `label == bucket_start` branch takes `features[labels <= label]`. There are")
    print(f"  {n_neg} sub-zero rows, and they are all swept into bucket 0 together with the rows at")
    print(f"  0.0 C and the rows int() truncates to 0, {b0_raw} in all, while the smoothing kernel")
    print("  is left to operate on a histogram that is wrong across half its range.")


def weighting_concentration():
    """Compare how concentrated the LDS weights get with and without the count clip."""
    import csv, numpy as np
    from common import bin_index, NBINS
    from scipy.ndimage import convolve1d
    from vendor.utils import get_lds_kernel_window
    temps = [float(r["temperature"]) for r in csv.DictReader(open("data/splits_iid.csv"))
             if r["split"] == "train"]
    counts = np.zeros(NBINS)
    for t in temps: counts[bin_index(t)] += 1
    kern = get_lds_kernel_window("gaussian", 5, 2)
    print("\nWEIGHT CONCENTRATION under inverse frequency")
    for name, clip in (("unclipped", None), ("clipped 5..1000", (5, 1000))):
        base = counts if clip is None else np.clip(counts, *clip)
        sm = convolve1d(base, weights=kern, mode="constant")
        w = np.array([1.0 / sm[bin_index(t)] if sm[bin_index(t)] > 0 else 0.0 for t in temps])
        w = w / w.mean()
        top = np.sort(w)[::-1][: max(1, round(0.01 * len(w)))].sum() / w.sum()
        print(f"  {name:16s} ratio max/min {w.max() / w[w > 0].min():>7.1f}   "
              f"heaviest 1 percent carry {100 * top:>4.1f} percent of the weight")


if __name__ == "__main__":
    bug1_keyerror(); bug2_truncation(); bug3_fds_labels(); weighting_concentration()
