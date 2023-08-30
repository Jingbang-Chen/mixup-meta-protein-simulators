# mixup-meta-protein-simulators
Official code release for the paper "Mixup-Augmented Meta-Learning for Sample-Efficient Fine-Tuning of Protein Simulators".
[![ArXiv](http://img.shields.io/badge/cs.LG-arXiv%3A2305.01140-B31B1B.svg)](https://arxiv.org/abs/2308.15116)

## Environment
Install the required packages from `requirements.txt`.
## Get Data
Run `collect_diff_temp_chignolin.py` to get the trajectories under different temperature. 

## Train the Model

### First-stage Pre-training
Run `pre_train_mix.py`.

### Second-stage Pre-training
Run `meta_train.py`.

### Evaluate the Model
Run `test_meta.py`.

