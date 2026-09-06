#!/usr/bin/env python3
"""Train one arm on a SkyFinder split and write its test table to JSON.

The checkpoint is picked by best validation MAE and then scored once on test.
"""
import argparse, csv, hashlib, json, os, random, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
from common import BIN_WIDTH, NBINS, bin_index, shot_fn, dsr_table, format_table, mae, frozen_bn
from vendor.resnet import resnet18
from vendor.loss import weighted_l1_loss
from vendor.utils import get_lds_kernel_window


def lds_weights(temps, reweight="sqrt_inv", kernel="gaussian", ks=5, sigma=2, smooth=True):
    """Per-sample target weights, following _prepare_weights in upstream imdb-wiki-dir/datasets.py
    step for step: transform the counts, sqrt for sqrt_inv or the 5..1000 clip for inverse,
    convolve with the kernel, invert, rescale to mean one. Two departures: bin_index replaces the
    int() lookup so signed temperatures stay in range, and the counts are floats throughout,
    whereas upstream's inverse path clips integer counts and convolve1d keeps that integer dtype;
    no stored result uses inverse.

    smooth=True is LDS: transform, convolve, invert. smooth=False is the reweighting alone, which
    is what separates SQINV from SQINV + LDS.
    """
    from scipy.ndimage import convolve1d
    assert reweight in ("none", "inverse", "sqrt_inv")
    if reweight == "none": return None
    counts = np.zeros(NBINS)
    for t in temps: counts[bin_index(t)] += 1
    # Transform the counts before smoothing, then invert after.
    if reweight == "sqrt_inv":
        counts = np.sqrt(counts)
    else:
        counts = np.clip(counts, 5, 1000)
    smoothed = (convolve1d(counts, weights=get_lds_kernel_window(kernel, ks, sigma), mode="constant")
                if smooth else counts)
    w = np.array([1.0 / smoothed[bin_index(t)] if smoothed[bin_index(t)] > 0 else 0.0
                  for t in temps], dtype=np.float32)
    return w * (len(w) / w.sum())


def load(split_csv, cache_npy, index_csv):
    idx = {(r["camera"], r["filename"]): i
           for i, r in enumerate(csv.DictReader(open(index_csv)))}
    X = np.load(cache_npy, mmap_mode="r")
    parts = {"train": [], "val": [], "test": []}
    for r in csv.DictReader(open(split_csv)):
        k = (r["camera"], r["filename"])
        if k in idx:
            parts[r["split"]].append({"i": idx[k], "camera": r["camera"],
                                      "filename": r["filename"],
                                      "temperature": float(r["temperature"])})
    return X, parts


def tensors(X, rows):
    a = np.ascontiguousarray(X[[r["i"] for r in rows]])
    # NHWC to NCHW, made contiguous, then normalised to [-1, 1] with mean 0.5 and std 0.5, the
    # reference transform.
    t = torch.from_numpy(a).permute(0, 3, 1, 2).contiguous().float().div_(255).sub_(0.5).div_(0.5)
    y = torch.tensor([[r["temperature"]] for r in rows], dtype=torch.float32)
    b = torch.tensor([[bin_index(r["temperature"])] for r in rows], dtype=torch.long)
    return t, y, b


@torch.no_grad()
def predict(model, Xt, dev, bs=128):
    model.eval()
    out = []
    for i in range(0, len(Xt), bs):
        out.append(model(Xt[i:i + bs].to(dev)).cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--arm", choices=["erm", "sqinv", "lds", "fds", "lds_fds"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--cache", default="data/cache_112.npy")
    ap.add_argument("--index", default="data/cache_112_index.csv")
    # "bin" is the DIR definition of the shot regions.
    ap.add_argument("--basis", choices=["group", "bin"], default="bin")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=("mps" if torch.backends.mps.is_available()
                                        else ("cuda" if torch.cuda.is_available() else "cpu")))
    ap.add_argument("--reweight", choices=["none", "inverse", "sqrt_inv"], default="sqrt_inv")
    ap.add_argument("--start-smooth", type=int, default=1,
                    help="epoch at which FDS begins calibrating features; beyond --epochs it "
                         "collects statistics without applying them.")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = a.device
    reweighted = a.arm in ("sqinv", "lds", "lds_fds")
    smooth_w = a.arm in ("lds", "lds_fds")          # sqinv reweights without the LDS kernel
    use_fds = a.arm in ("fds", "lds_fds")

    X, parts = load(a.splits, a.cache, a.index)
    tr, va, te = parts["train"], parts["val"], parts["test"]
    Xtr, ytr, btr = tensors(X, tr)
    Xva, yva, _ = tensors(X, va)
    Xte, yte, _ = tensors(X, te)
    if reweighted:
        wv = lds_weights([r["temperature"] for r in tr], reweight=a.reweight, smooth=smooth_w)
        if wv is None:
            sys.exit(f"--arm {a.arm} needs a reweighting scheme. --reweight none produces no "
                     "weights; use --arm erm for an unweighted run.")
        w = torch.from_numpy(wv).unsqueeze(1)
        print(f"  {'LDS' if smooth_w else 'reweight-only'} [{a.reweight}]: "
              f"weight min {wv.min():.3f} max {wv.max():.3f} "
              f"ratio {wv.max()/wv.min():.1f}")
    else:
        w = torch.ones(len(tr), 1)

    # Restrict the FDS buckets to the occupied range of the training labels, so the empty
    # leading and trailing buckets are not smoothed into the real ones at the edges.
    occupied = sorted({bin_index(r["temperature"]) for r in tr})
    b_start, b_num = occupied[0], occupied[-1] + 1
    if use_fds:
        print(f"  FDS buckets: start {b_start}, num {b_num} "
              f"({len(occupied)} occupied of {b_num - b_start} in range; "
              f"NBINS={NBINS}, so {b_start} leading and {NBINS - b_num} trailing empties excluded)")
    # ks=5, sigma=2 are the reference README's LDS and FDS settings; start_update 0,
    # start_smooth 1 and momentum 0.9 are the reference train.py defaults.
    model = resnet18(fds=use_fds, bucket_num=b_num, bucket_start=b_start, start_update=0,
                     start_smooth=a.start_smooth, kernel="gaussian", ks=5, sigma=2,
                     momentum=0.9).to(dev)
    if use_fds and a.start_smooth > a.epochs:
        print(f"  DIAGNOSTIC: start_smooth={a.start_smooth} > epochs={a.epochs}; FDS collects "
              f"statistics and never calibrates. See run_all.sh for what a match with ERM shows.")
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    N = len(tr); best = (float("inf"), -1, None); t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(N)
        tot = 0.0
        for i in range(0, N, a.batch):
            s = perm[i:i + a.batch]
            xb, yb, bb, wb = (Xtr[s].to(dev), ytr[s].to(dev), btr[s].to(dev), w[s].to(dev))
            o = model(xb, bb, ep)
            pred = o[0] if isinstance(o, tuple) else o
            loss = weighted_l1_loss(pred, yb, wb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(s)
        if use_fds:
            # Collect the epoch's features for FDS. This has to stay in train() mode, since
            # that is what makes forward return the encoding, but BatchNorm is held in eval()
            # so the pass does not re-estimate its running statistics.
            model.train()
            encs, labs = [], []
            with frozen_bn(model), torch.no_grad():
                for i in range(0, N, 128):
                    _, f = model(Xtr[i:i + 128].to(dev), btr[i:i + 128].to(dev), ep)
                    encs.append(f.detach()); labs.append(btr[i:i + 128].squeeze(1).to(dev))
            model.FDS.update_last_epoch_stats(ep)
            model.FDS.update_running_stats(torch.cat(encs), torch.cat(labs), ep)
        sched.step()
        vmae = mae([(p - y).item() for p, y in zip(predict(model, Xva, dev), yva)])
        if vmae < best[0]:
            best = (vmae, ep, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        print(f"  ep {ep:>3}  train_loss {tot/N:7.4f}  val_MAE {vmae:6.3f}"
              f"{'  <- best' if ep == best[1] else ''}", flush=True)

    model.load_state_dict(best[2])
    errs = [(p - y).item() for p, y in zip(predict(model, Xte, dev), yte)]
    region, _ = shot_fn(tr, BIN_WIDTH, basis=a.basis)
    tbl = dsr_table(errs, te, region)
    print(f"\narm={a.arm} seed={a.seed} split={os.path.basename(a.splits)} "
          f"best_epoch={best[1]} val_MAE={best[0]:.3f} wall={time.time()-t0:.0f}s")
    print(format_table(tbl, "TEST"))

    out = a.out or f"results/{os.path.basename(a.splits).replace('.csv','')}_{a.arm}_s{a.seed}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # Write to a temp file and rename, so an interrupted run leaves no half-written JSON.
    tmp = out + ".partial"
    with open(tmp, "w") as _fh:
        json.dump({"arm": a.arm, "seed": a.seed, "splits": a.splits, "basis": a.basis,
                   "reweight": a.reweight if reweighted else "none",
                   "env": f"py{sys.version_info.major}.{sys.version_info.minor}."
                          f"{sys.version_info.micro}-torch{torch.__version__}",
                   # hash the split file so results from a rebuilt split of the same name
                   # are not pooled with these
                   "split_sha256": hashlib.sha256(open(a.splits, "rb").read()).hexdigest(),
                   "start_smooth": a.start_smooth,
                   "fds_bn_frozen": True if use_fds else None,
                   "fds_bucket_start": b_start if use_fds else None,
                   "fds_bucket_num": b_num if use_fds else None,
                   "epochs": a.epochs, "lr": a.lr, "batch": a.batch,
                   "best_epoch": best[1], "val_mae": best[0],
                   "wall_s": round(time.time() - t0, 1),
                   "n_train": len(tr), "n_val": len(va), "n_test": len(te),
                   "table": {k: v for k, v in tbl.items() if k != "_worst_attr"},
                   "worst_attr": tbl.get("_worst_attr"),
                   # full precision, so the tables recompute exactly from the stored residuals
                   "errors": errs}, _fh, indent=1)
        _fh.flush(); os.fsync(_fh.fileno())
    os.replace(tmp, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
