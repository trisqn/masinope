---
dataset_info:
  features:
  - name: image
    dtype: image
  - name: age
    dtype: uint8
  - name: gender
    dtype: uint8
  splits:
  - name: train
    num_bytes: 7426813301.0
    num_examples: 39610
  download_size: 7423712808
  dataset_size: 7426813301.0
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
