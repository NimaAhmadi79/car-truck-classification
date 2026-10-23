# Data

This project uses **CIFAR-10**, filtered to its two road-vehicle classes:
`automobile` (CIFAR-10 class 1) → **car (label 0)** and `truck` (CIFAR-10 class 9) →
**truck (label 1)**. Filtering CIFAR-10's 10 classes down to these two yields **10,000
training** and **2,000 test** images.

CIFAR-10 is a public dataset released by the Canadian Institute for Advanced Research
(Krizhevsky, 2009), distributed for research use. The full dataset is **not committed**
here — `torchvision` downloads it automatically on first run.

## What is in this folder

| File / folder                | Committed? | What it is                                                        |
| ---------------------------- | :--------: | ----------------------------------------------------------------- |
| `test_sample.pt`             | ✅ yes     | Small convenience subset: **200** car/truck test images + labels  |
| `train_sample.pt`            | ⚠️ see below | Small convenience subset: **1,000** car/truck train images + labels |
| `cifar-10-batches-py/`       | ❌ no      | Full CIFAR-10, created by `torchvision` on download                |
| `cifar-10-python.tar.gz`     | ❌ no      | Full CIFAR-10 archive (~170 MB), downloaded by `torchvision`       |

### About the sample `.pt` files

`train_sample.pt` and `test_sample.pt` each hold a dict `{"images": FloatTensor,
"labels": LongTensor}`. Images are `(N, 3, 32, 32)` float tensors in `[0, 1]`; labels
are `0` (car) / `1` (truck). They exist so the notebook can run **quickly on a small
subset** without downloading all of CIFAR-10.

> **Important — which numbers are the real ones.** When these sample files are present,
> the notebook loads them *instead of* the full dataset. Evaluating on the 200-image
> `test_sample.pt` gives near-perfect scores that are **not** representative. The
> headline results in the main README (ResNet18 97.1%, Swin-Tiny 96.2%, SimpleCNN
> 90.2%) come from the **full 2,000-image** test set. To reproduce those, remove or
> rename the sample files so the notebook downloads and uses full CIFAR-10.

### `train_sample.pt` may be missing

`train_sample.pt` is ~12 MB. Depending on how this repository was populated it may not
be present (the tool used to build the repo could not transfer files larger than
10 MB). If it is missing, you have two easy options:

1. **Just run the notebook on full CIFAR-10** — if neither sample file exists, the
   notebook downloads CIFAR-10 and filters it automatically. This is the path that
   reproduces the reported results.
2. **Regenerate the sample files** from full CIFAR-10 with a short script (filter
   classes 1 and 9, take the first N of each, and `torch.save({"images":..,
   "labels":..})`).

## How the notebook finds the data

The **Environment setup** cell defines `DATA_DIR`:

- **Colab:** `MyDrive/Colab_Notebooks/Deep_Learning_Project_Final/data`
- **Local:** `./data` by default, or set `CAR_TRUCK_PROJECT_DIR` to point elsewhere.

CIFAR-10 is downloaded into `DATA_DIR`, and the sample `.pt` files are read from there.

## License / attribution

CIFAR-10 © its original authors; see
<https://www.cs.toronto.edu/~kriz/cifar.html>. This repository does not redistribute
the full dataset; the small sample tensors are derived from it for convenience and
carry the same original terms.
