# Model checkpoints

Trained weights for the three models compared in this project. Each `.pth` is a PyTorch
`state_dict` saved with `torch.save(model.state_dict(), ...)` and loaded with
`model.load_state_dict(torch.load(path, map_location=device))`.

| File                          | Committed? | Size    | Architecture                                   |
| ----------------------------- | :--------: | ------- | ---------------------------------------------- |
| `simple_cnn_car_truck.pth`    | ✅ yes     | ~1.4 MB | From-scratch `SimpleCNN` baseline              |
| `resnet18_car_truck.pth`      | ⚠️ see below | ~45 MB | ResNet-18 (ImageNet-pretrained, fine-tuned)    |
| `swin_tiny_car_truck.pth`     | ❌ no      | ~110 MB | Swin-Tiny (ImageNet-pretrained, fine-tuned)    |

## Which checkpoints are here

- **`simple_cnn_car_truck.pth`** is committed — it is small and lets you reproduce the
  baseline's inference immediately.
- **`resnet18_car_truck.pth`** (~45 MB) may be missing: the tool used to assemble this
  repository could not transfer files larger than 10 MB. It fits within GitHub's limits,
  so it can be added later (see below).
- **`swin_tiny_car_truck.pth`** (~110 MB) is intentionally **not** committed — it
  exceeds GitHub's 100 MB per-file hard limit and would require Git LFS. Regenerate it
  by running the notebook.

## Regenerating any missing checkpoint

Run `car_truck_modeling.ipynb` top to bottom. The training cells rebuild and save all
three checkpoints into the `models/` directory (`MODELS_DIR` in the notebook). The
ResNet-18 and Swin-Tiny fine-tuning runs are short (5 epochs each); SimpleCNN trains
for 10 epochs.

## Adding `resnet18_car_truck.pth` manually

If you have the file (e.g. from the original Google Drive), drop it into this folder:

```
models/resnet18_car_truck.pth
```

It will then be picked up by the notebook's loading cell and by any inference script.

## Loading a checkpoint for inference

```python
import torch
from torchvision.models import resnet18

device = "cuda" if torch.cuda.is_available() else "cpu"

model = resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)   # 2 classes: car / truck
model.load_state_dict(torch.load("models/resnet18_car_truck.pth", map_location=device))
model.eval().to(device)
```

The `SimpleCNN` baseline is defined in the notebook; instantiate that class before
loading `simple_cnn_car_truck.pth`. Label mapping is `0 = car`, `1 = truck`.
