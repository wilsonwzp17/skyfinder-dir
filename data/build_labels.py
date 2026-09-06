#!/usr/bin/env python3
"""Turn the released SkyFinder metadata table into a clean labels file."""
import argparse, csv, os

SENTINEL = -9998.0  # TempM carries -9999.0 when the reading is missing

# Cameras whose archive did not fetch from the project-page route this pipeline uses. That is a
# property of the route taken here, not a claim that the data is unavailable elsewhere.
NO_ARCHIVE = {"8887", "21444", "21510", "21511", "22788", "24325"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/complete_table_with_mcr.csv")
    ap.add_argument("--images", default=None, help="dir of <camera>/<file>.jpg; restricts to what exists")
    ap.add_argument("--out", default="data/labels.csv")
    ap.add_argument("--require-zip", action="store_true",
                    help="drop the 6 cameras whose archive did not fetch here, leaving the 47 this pipeline uses")
    a = ap.parse_args()

    on_disk = None
    if a.images and os.path.isdir(a.images):
        on_disk = set()
        for cam in os.listdir(a.images):
            d = os.path.join(a.images, cam)
            if os.path.isdir(d):
                for fn in os.listdir(d):
                    if fn.lower().endswith(".jpg"):
                        on_disk.add((cam, fn))
        print(f"images on disk: {len(on_disk)} across {len({c for c, _ in on_disk})} cameras")

    kept = []
    drop_sent = drop_missing = drop_noarch = 0
    seen_cams = set()
    for r in csv.DictReader(open(a.csv)):
        seen_cams.add(r["CamId"])
        try:
            t = float(r["TempM"])
        except (TypeError, ValueError):
            drop_sent += 1
            continue
        if t <= SENTINEL:
            drop_sent += 1
            continue
        if a.require_zip and r["CamId"] in NO_ARCHIVE:
            drop_noarch += 1
            continue
        key = (r["CamId"], r["Filename"])
        if on_disk is not None and key not in on_disk:
            drop_missing += 1
            continue
        kept.append((r["Filename"], r["CamId"], f"{t:.1f}"))
    print(f"cameras in table: {len(seen_cams)}")
    print(f"dropped: sentinel/unparseable TempM {drop_sent}; no archive {drop_noarch}; no image on disk {drop_missing}")
    print(f"kept: {len(kept)} rows across {len({c for _, c, _ in kept})} cameras")
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "camera", "temperature"])
        w.writerows(kept)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
