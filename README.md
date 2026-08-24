# Animal Image Classifiers

Two image classification models built from scratch in PyTorch, trained on Google Colab:

1. **Binary classifier** — cat vs. dog
2. **Multi-class classifier** — cat vs. dog vs. bird vs. other

Both are small Convolutional Neural Networks (CNNs) trained via supervised learning, with data augmentation and best-checkpoint saving

---

## 1. Binary Model — Cat vs. Dog

### Dataset
- **Source:** Kaggle — [`tongpython/cat-and-dog`](https://www.kaggle.com/datasets/tongpython/cat-and-dog)
- **Size:** ~4,000 cats + ~4,000 dogs (train), ~1,000 each (test)
- **Split:** 80% train / 20% validation (from the training folder) + separate held-out test set

### Preprocessing
- Resized to **128×128**
- Normalized with ImageNet mean/std: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`
- **Training augmentation:** random horizontal flip, random rotation (±15°), color jitter (brightness/contrast)
- Validation/test: no augmentation

### Model architecture — `CatDogCNN`
```
Conv2d(3→16) → ReLU → MaxPool
Conv2d(16→32) → ReLU → MaxPool
Conv2d(32→64) → ReLU → MaxPool
Flatten → Linear(16384→128) → ReLU → Dropout(0.3)
Linear(128→2)
```

### Training
- Loss: `CrossEntropyLoss`
- Optimizer: `Adam`, lr=0.001
- 15 epochs, batch size 32
- Best model checkpointed on validation accuracy (not just the last epoch)

### Results
| Version | Best Val Accuracy | Notes |
|---|---|---|
| No augmentation | 78.6% | Overfit after ~epoch 5 (train acc hit 98.6%, val loss rose) |
| With augmentation | **81.4%** | Train/val gap stayed healthy, no runaway overfitting |


### Files
- `cat_dog_model.pth` — checkpoint containing `model_state_dict`, `class_names`, `image_size`, `val_accuracy`

### Running predictions locally (no retraining needed)
```bash
pip install torch torchvision pillow matplotlib
```
```python
checkpoint = torch.load("cat_dog_model.pth", map_location=device)
model = CatDogCNN().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```
Preprocessing at inference time must exactly match training (resize 128×128 + same normalization).

---

## 2. Multi-Class Model — Cat / Dog / Bird / Other

### Dataset
Assembled from two Kaggle sources:
- **Animals:** [`mahmoudnoor/high-resolution-catdogbird-image-dataset-13000`](https://www.kaggle.com/datasets/mahmoudnoor/high-resolution-catdogbird-image-dataset-13000) — cat, dog, bird
- **Other:** [`puneet6060/intel-image-classification`](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) — 6 non-animal scene categories (buildings, forest, glacier, mountain, sea, street)

### Class balancing
Raw counts were imbalanced (cat: 4,015 / dog: 5,180 / bird: 4,149 / other: 14,034 if using all scene images). The "other" class was **undersampled to ~4,500**, drawn proportionally (750 images) across all 6 scene categories to keep it diverse rather than dominated by one scene type.

Final dataset: **cat: 4,015 · dog: 5,180 · bird: 4,149 · other: 4,500**

### Preprocessing
Same as the binary model (128×128, ImageNet normalization, same augmentation strategy), with a **70/15/15 train/val/test split** (this dataset had no pre-made test folder, so the split was built manually with a fixed random seed for reproducibility).

### Model architecture — `MultiAnimalCNN`
Identical backbone to the binary model — only the output layer changed:
```
Conv2d(3→16) → ReLU → MaxPool
Conv2d(16→32) → ReLU → MaxPool
Conv2d(32→64) → ReLU → MaxPool
Flatten → Linear(16384→128) → ReLU → Dropout(0.3)
Linear(128→4)   # cat, dog, bird, other
```
This demonstrates that the convolutional feature extractor generalizes across tasks — only the final classification head needed to change for more classes.

### Training
Same setup as the binary model (Adam, lr=0.001, 15 epochs, batch 32, best-checkpoint saving from the start).

### Results
- **Best validation accuracy:** 81.8% (epoch 12)
- **Test accuracy:** 80.5%

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| bird | 0.81 | 0.86 | 0.83 | 628 |
| cat | 0.69 | 0.69 | 0.69 | 584 |
| dog | 0.75 | 0.68 | 0.71 | 747 |
| other | 0.94 | 0.98 | 0.96 | 719 |

**Confusion matrix:**
|  | pred: bird | pred: cat | pred: dog | pred: other |
|---|---|---|---|---|
| **actual: bird** | 543 | 27 | 38 | 20 |
| **actual: cat** | 43 | 403 | 130 | 8 |
| **actual: dog** | 77 | 149 | 508 | 13 |
| **actual: other** | 10 | 3 | 4 | 702 |

### Files
- `best_multi_model.pth` — best checkpoint (state dict only)
- Class order (alphabetical, from `ImageFolder`): `['bird', 'cat', 'dog', 'other']`

### Running predictions locally (no retraining needed)
Same process as the binary model — reload `MultiAnimalCNN(num_classes=4)`, load the state dict, use the same 128×128 + normalization preprocessing, and apply `softmax` to get per-class probabilities.