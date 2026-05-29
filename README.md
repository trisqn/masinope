# Age Prediction Model

TensorFlow/Keras age regression model using MobileNetV2.

By default, training loads the Hugging Face dataset:

```python
from datasets import load_dataset

ds = load_dataset("typorch/age")
```

The current training setup filters the dataset to ages `0 <= age < 60`, so this is a `0-60` age predictor. Predictions are clamped to that range.

## Install

Use Python 3.11:

```powershell
py -3.11 -m pip install tensorflow datasets pillow numpy
```

## Train

```powershell
py -3.11 main.py --device gpu --epochs 16 --fine-tune-epochs 6 --patience 5
```

If ImageNet weights cannot be downloaded:

```powershell
py -3.11 main.py --device gpu --weights none
```

Useful options:

```powershell
py -3.11 main.py --device cpu
py -3.11 main.py --data-dir age/data
py -3.11 main.py --batch-size 32
py -3.11 main.py --no-use-sample-weights
py -3.11 main.py --sample-weight-cap 8 --sample-weight-power 1.15
```

Training outputs are written to `model_out/`, including:

- `best_age_model.keras`
- `last_age_model.keras`
- `training_log.csv`
- `validation_predictions.csv`
- `validation_age_bin_metrics.csv`
- `test_predictions.csv`
- `test_age_bin_metrics.csv`
- `summary.json`

## Predict

Run on one image:

```powershell
py -3.11 predict.py C:\path\to\face.jpg
```

Run on a folder:

```powershell
py -3.11 predict.py C:\path\to\images
```

Save predictions to CSV:

```powershell
py -3.11 predict.py C:\path\to\images --csv predictions.csv
```

`predict.py` uses `model_out\best_age_model.keras` by default.

## Current Result

Latest improved run:

- Validation MAE: `6.5380` years
- Test MAE: `6.6056` years
- Test RMSE: `9.2369` years

Per-bin test MAE:

- `0-9`: `3.3719`
- `10-19`: `7.3150`
- `20-29`: `6.4579`
- `30-39`: `7.2798`
- `40-49`: `12.3165`
- `50-59`: `14.3414`

The model is still weakest in the `40-59` range because those bins have far fewer examples than younger ages.
