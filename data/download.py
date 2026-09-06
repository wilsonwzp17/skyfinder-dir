#!/usr/bin/env python3
"""Fetch the metadata table and the twelve camera archives, and unpack them to
images/<camera>/<basename>.jpg."""
import argparse, concurrent.futures as cf, os, shutil, subprocess, sys, zipfile

BASE = "https://cs.valdosta.edu/~rpmihail/skyfinder/images/{}.zip"
# The table of temperature labels. As of September 2026 the project page's own link to it fails;
# this path serves it.
TABLE = "https://cs.valdosta.edu/~rpmihail/skyfinder/analysis/complete_table_with_mcr.csv"
TABLE_BYTES = 92389332
# The twelve smallest of the 47 fetchable archives by Content-Length, ascending; HEAD-probe BASE
# to check.
TWELVE = ["858", "3888", "4795", "5021", "1093", "3297", "4232", "4801",
          "19106", "8438", "17218", "9730"]


def fetch(cam, raw_dir):
    dst = os.path.join(raw_dir, f"{cam}.zip")
    if os.path.exists(dst) and zipfile.is_zipfile(dst):
        return cam, os.path.getsize(dst), "cached"
    tmp = dst + ".part"
    r = subprocess.run(["curl", "-sSL", "--max-time", "600", "-o", tmp, BASE.format(cam),
                        "-w", "%{http_code}"], capture_output=True, text=True)
    code = (r.stdout or "").strip()
    if code != "200" or not zipfile.is_zipfile(tmp):
        if os.path.exists(tmp): os.remove(tmp)
        return cam, 0, f"FAILED http {code}"
    os.rename(tmp, dst)
    return cam, os.path.getsize(dst), "ok"


def fetch_table(dst="data/complete_table_with_mcr.csv"):
    if os.path.exists(dst) and os.path.getsize(dst) == TABLE_BYTES:
        print(f"  table    cached  {os.path.getsize(dst)/1e6:.1f} MB"); return
    tmp = dst + ".part"
    r = subprocess.run(["curl", "-sSL", "--max-time", "1800", "-o", tmp, TABLE,
                        "-w", "%{http_code}"], capture_output=True, text=True)
    code = (r.stdout or "").strip()
    size = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    if code != "200" or size != TABLE_BYTES:
        if os.path.exists(tmp): os.remove(tmp)
        sys.exit(f"metadata table download failed: http {code}, {size} bytes, "
                 f"expected {TABLE_BYTES}")
    os.replace(tmp, dst)
    print(f"  table    ok      {size/1e6:.1f} MB")


def unpack(cam, raw_dir, img_dir):
    out = os.path.join(img_dir, cam)
    if os.path.isdir(out) and os.listdir(out):
        return cam, len(os.listdir(out)), "cached"
    os.makedirs(out, exist_ok=True)
    n = 0
    with zipfile.ZipFile(os.path.join(raw_dir, f"{cam}.zip")) as z:
        for m in z.namelist():
            if not m.lower().endswith(".jpg"): continue
            base = os.path.basename(m)          # flatten the archive's internal directory path
            with z.open(m) as src, open(os.path.join(out, base), "wb") as dst:
                shutil.copyfileobj(src, dst)
            n += 1
    return cam, n, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", default=",".join(TWELVE))
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--images", default="data/images")
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    cams = a.cameras.split(",")
    os.makedirs(a.raw, exist_ok=True); os.makedirs(a.images, exist_ok=True)
    os.makedirs("data", exist_ok=True)
    fetch_table()

    total = 0
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        for cam, size, st in ex.map(lambda c: fetch(c, a.raw), cams):
            total += size
            print(f"  fetch {cam:>6}  {size/1e6:8.1f} MB  {st}")
            if st.startswith("FAILED"): sys.exit(f"download failed for {cam}")
    print(f"downloaded {total/1e6:.1f} MB\n")

    grand = 0
    for cam in cams:
        cam, n, st = unpack(cam, a.raw, a.images)
        grand += n
        print(f"  unpack {cam:>6}  {n:>6} jpg  {st}")
    print(f"\n{grand} images under {a.images}/<camera>/<basename>.jpg")


if __name__ == "__main__":
    main()
