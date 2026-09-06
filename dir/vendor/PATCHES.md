# Vendored from YyzHarry/imbalanced-regression, `imdb-wiki-dir/`, commit `a6fdc45`

**Licence.** The upstream project is MIT, Copyright (c) 2021 Yuzhe Yang. Its full notice is kept
beside these files in [LICENSE](LICENSE), as the MIT terms require for substantial portions. The
files in this directory are the upstream author's work with the edits listed below. Licensing for
the rest of the repository is stated in the [root LICENSE](../../LICENSE).

Kept here so the diff against upstream is inspectable.


| File | Status | Change | Why |
|---|---|---|---|
| `loss.py` | **verbatim** | none | works as shipped |
| `utils.py` | **verbatim** | none | works as shipped |
| `fds.py` | modified, 2 edits | `_get_kernel_window` returned `torch.tensor(...).cuda()`; the `.cuda()` is dropped and the window is registered as a buffer so it moves with `.to(device)` | raises `AssertionError: Torch not compiled with CUDA enabled` at construction on this machine |
| `resnet.py` | modified, 3 edits | `nn.AvgPool2d(7, stride=1)` becomes `nn.AdaptiveAvgPool2d(1)`; `x.view(...)` becomes `x.reshape(...)`; a `resnet18` factory is appended | `AvgPool2d(7)` assumes a 224px input and throws at 112px, where layer4 is 4x4; the `.reshape()` edit was made while chasing an MPS backward error whose message recommended `.reshape`, and the real cause was a non-contiguous input tensor from my own cache, fixed by `.contiguous()` in `dir/train.py` `tensors()`; on that contiguous input `.view()` and `.reshape()` give identical outputs and gradients, so this edit changes no number and is kept as a harmless precaution; upstream defines only `resnet50` |

Three further points that are **not** patches but are required to use this correctly:

1. **FDS must be fed integer bin indices, not raw temperatures.** `update_running_stats` and `smooth` index with `int(label - bucket_start)`, so passing a signed float temperature both mis-buckets and sweeps every negative value into bucket 0. We pass `common.bin_index(t)`.
2. **`_prepare_weights` in upstream `datasets.py` is not called.** It builds `{x: 0 for x in range(max_target)}` and indexes it with `min(max_target-1, int(label))`, which raises `KeyError` on a negative temperature, and, once a constant bin shift stops the crash, `int()` truncates toward zero so the bin holding zero would be double width. `lds_weights()` in `dir/train.py` follows the same steps, with `common.bin_index` in place of the `int()` lookup.
3. **The training loop in `dir/train.py` is not vendored.** It keeps the reference order for FDS, `update_last_epoch_stats` then `update_running_stats` after the gradient pass, and departs from the reference `train.py` at the epoch-end collection pass, which upstream runs with BatchNorm in train mode, lines 244 and 269 to 281 at this commit; here `common.frozen_bn` holds BatchNorm in `eval()` for that pass so the no-smoothing control matches ERM exactly. The data path, optimiser, schedule and checkpointing are also not vendored.
