#!/usr/bin/env python3
"""Pins down what the vendored FDS keeps in its *_last_epoch buffers, since calibrate_mean_var
consumes the raw and the smoothed pair together."""
import os, sys, torch
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))
sys.path.insert(0, os.path.join(_here, "vendor"))   # vendored files import each other flatly
from vendor.fds import FDS


def main():
    f = FDS(feature_dim=3, bucket_num=6, bucket_start=0, start_update=0, start_smooth=1, momentum=0.9)
    labs = torch.randint(0, 6, (30,))
    f.update_running_stats(torch.randn(30, 3), labs, 0)
    f.update_last_epoch_stats(1)

    aliased = f.running_mean_last_epoch.data_ptr() == f.running_mean.data_ptr()
    print(f"  running_mean_last_epoch shares storage with running_mean: {aliased}")
    assert aliased, "upstream behaviour changed: the buffers no longer alias"

    smoothed_before = f.smoothed_mean_last_epoch.clone()
    f.update_running_stats(torch.randn(30, 3) * 3 + 5, labs, 1)
    tracks = torch.equal(f.running_mean_last_epoch, f.running_mean)
    frozen = torch.equal(f.smoothed_mean_last_epoch, smoothed_before)
    print(f"  after an update, the RAW last-epoch buffer tracks the new statistics: {tracks}")
    print(f"  after an update, the SMOOTHED last-epoch buffer stays frozen:        {frozen}")
    assert tracks and frozen

    # resnet.py binds `encoding_s = encoding` and smooth() writes into its argument in place,
    # so once smoothing starts the encoding the forward pass returns is the smoothed one.
    import inspect
    from vendor import resnet as _rn
    src = inspect.getsource(_rn.ResNet.forward)
    aliased_enc = "encoding_s = encoding" in src
    # check the in-place write by running it rather than by reading the source
    f2 = FDS(feature_dim=3, bucket_num=6, bucket_start=0, start_update=0, start_smooth=0, momentum=None)
    f2.update_running_stats(torch.randn(40, 3), torch.randint(0, 6, (40,)), 0)
    f2.update_last_epoch_stats(1)
    enc = torch.randn(10, 3)
    enc_s = enc                      # exactly what resnet.forward does
    before = enc.clone()
    f2.smooth(enc_s, torch.randint(0, 6, (10, 1)), 1)
    mutated = not torch.equal(enc, before)
    print(f"\n  smooth() writes into the tensor it was handed: {mutated}")
    assert mutated, "smooth() no longer mutates in place; re-check what the collection pass sees"
    print(f"  resnet.forward binds encoding_s to encoding without copying: {aliased_enc}")
    print("  Together: once smoothing starts, the tensor the forward pass returns as the encoding")
    print("  is the smoothed one, so the epoch-end collection pass accumulates statistics from")
    print("  smoothed features rather than raw ones.")
    assert aliased_enc, "upstream forward changed; re-check what the collection pass consumes"

    print("\n  So during the next epoch calibrate_mean_var receives raw statistics from after the last")
    print("  update and smoothed statistics from before it, and once smoothing begins the running")
    print("  statistics are accumulated from smoothed features. Both behaviours are in the pinned")
    print("  upstream code, a6fdc45, and are left in place; see the implementation section")
    print("  of report/REPORT.md.")


if __name__ == "__main__":
    main()
