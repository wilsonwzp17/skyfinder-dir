#!/usr/bin/env python3
"""Decode every labelled image once into cache_<res>.npy, uint8 NHWC, with cache_<res>_index.csv
giving the row order as (camera, filename, temperature)."""
import argparse, concurrent.futures as cf, csv, os, sys
import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--images", default="data/images")
    ap.add_argument("--res", type=int, default=112)
    ap.add_argument("--out", default="data/cache")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()

    rows = [(r["camera"], r["filename"], float(r["temperature"]))
            for r in csv.DictReader(open(a.labels))]
    # truncated JPEGs are skipped here rather than decoded with PIL's truncated-load flag
    corrupt = set()
    if os.path.exists("data/corrupt_images.csv"):
        corrupt = {(r["camera"], r["filename"])
                   for r in csv.DictReader(open("data/corrupt_images.csv"))}
    keep, missing, dropped = [], 0, 0
    for cam, fn, t in rows:
        if not os.path.exists(os.path.join(a.images, cam, fn)): missing += 1; continue
        if (cam, fn) in corrupt: dropped += 1; continue
        keep.append((cam, fn, t))
    print(f"labelled rows {len(rows)}   on disk {len(keep)+dropped}   missing image {missing}")
    print(f"dropped as undecodable: {dropped}  (see data/corrupt_images.csv)")
    on_disk = sum(len(os.listdir(os.path.join(a.images, c)))
                  for c in os.listdir(a.images) if os.path.isdir(os.path.join(a.images, c)))
    print(f"images on disk {on_disk}   of which not carried into the cache "
          f"{on_disk - len(keep)}, either unlabelled or listed as corrupt")

    def load(i):
        cam, fn, _ = keep[i]
        im = Image.open(os.path.join(a.images, cam, fn)).convert("RGB").resize((a.res, a.res))
        return i, np.asarray(im, dtype=np.uint8)

    arr = np.zeros((len(keep), a.res, a.res, 3), dtype=np.uint8)
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        for i, im in ex.map(load, range(len(keep))):
            arr[i] = im
    np.save(f"{a.out}_{a.res}.npy", arr)
    with open(f"{a.out}_{a.res}_index.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["camera", "filename", "temperature"])
        for cam, fn, t in keep: w.writerow([cam, fn, f"{t:.1f}"])
    print(f"wrote {a.out}_{a.res}.npy  {arr.shape} {arr.nbytes/1e6:.0f} MB")
    print(f"wrote {a.out}_{a.res}_index.csv  ({len(keep)} rows)")
    ks = [(c, f) for c, f, _ in keep]
    assert len(ks) == len(set(ks)), "duplicate (camera, filename)"
    print("key integrity: (camera, filename) unique  OK")


if __name__ == "__main__":
    main()
