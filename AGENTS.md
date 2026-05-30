# Project Notes

## Age Prediction Model

- `main.py` is the training entry point for a TensorFlow/Keras age regression model based on MobileNetV2.
- The default dataset source is Hugging Face:

```python
from datasets import load_dataset

ds = load_dataset("typorch/age")
```

- `main.py` loads `typorch/age` by default, then creates stratified train/validation/test splits from the selected split.
- Local Hugging Face-style Parquet shards are still supported with `--data-dir age/data`.
- Each row must contain an `image` column and an `age` column.
- Images are decoded through Pillow before being passed to TensorFlow. This strips problematic PNG color profiles and avoids TensorFlow/libpng warnings such as `iCCP: known incorrect sRGB profile`.
- Training outputs are written to `model_out/`:
  - `best_age_model.keras`
  - `last_age_model.keras`
  - `training_log.csv`
  - `validation_predictions.csv`
  - `validation_age_bin_metrics.csv`
  - `test_predictions.csv`
  - `test_age_bin_metrics.csv`
  - `summary.json`

## Running

Use Python 3.11:

```powershell
py -3.11 -m pip install tensorflow datasets pillow numpy
py -3.11 main.py
```

Useful options:

```powershell
py -3.11 main.py --epochs 12 --fine-tune-epochs 3 --batch-size 32
py -3.11 main.py --weights none
py -3.11 main.py --data-dir age/data
py -3.11 main.py --device gpu
py -3.11 main.py --test-fraction 0.1
py -3.11 main.py --no-use-sample-weights
```

Next experiment to continue:

```powershell
py -3.11 main.py --device gpu --epochs 18 --fine-tune-epochs 24 --patience 8 --fine-tune-layers 90 --fine-tune-learning-rate 7.5e-6 --sample-weight-cap 6 --sample-weight-power 1.1
```

Current best overall run:

```powershell
py -3.11 main.py --device gpu --epochs 18 --fine-tune-epochs 24 --patience 8 --fine-tune-layers 90 --fine-tune-learning-rate 7.5e-6 --sample-weight-cap 8 --sample-weight-power 1.15
```

- Validation MAE: `5.8379`
- Test MAE: `5.8720`
- Note: this is the best overall MAE so far. The layer sweep favored `90` fine-tune layers, and `7.5e-6` fine-tune learning rate beat both `1e-5` and `5e-6` overall. Older bins remain weak (`40-49` test MAE `12.8242`, `50-59` test MAE `15.3517`), while `0-29` improved. Stronger weighting did not consistently improve older-bin test MAE, so the next experiment keeps the current best schedule and slightly reduces sample weighting to `cap=6`, `power=1.1`.

Use `--weights none` when ImageNet weights cannot be downloaded.

## Evaluation Notes

- The model is a convolutional neural network because it uses MobileNetV2 as the image backbone.
- The dataset is heavily age-imbalanced. The script splits by age buckets so validation/test data are not accidentally concentrated in one shard or age range.
- Training uses inverse-frequency sample weights by default, capped at `5.0`, so rare older-age buckets matter more during optimization.
- `summary.json` includes train/validation/test age distributions and validation/test metrics.
- The per-bin CSV metric files are important because a single overall MAE can hide poor performance on rare ages.

## GPU Notes

- The script prints visible TensorFlow GPU devices at startup.
- `--device gpu` now fails early if TensorFlow cannot see a GPU, instead of silently training on CPU.
- Native Windows TensorFlow 2.11+ does not use NVIDIA CUDA GPUs. For NVIDIA GPU training, use WSL2/Linux with a GPU-enabled TensorFlow install.
- A DirectML TensorFlow stack may be an option on native Windows, but it must be installed and verified separately.
