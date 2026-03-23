# CLABP

This repository contains a AMPs activity prediction pipeline based on a multimodal model (`ABP_Text_Picture_Model`) with contrastive learning.

## Overview

The project includes:

- Data preprocessing from PDB files
- Multimodal feature construction (sequence, phi/psi, DSSP, distance, movement, quaternion)
- Training with classification + contrastive loss
- Evaluation on independent datasets

Core scripts:

- `train_with_contrastive.py`: training and checkpointing
- `eval.py`: offline evaluation
- `model.py`: network definition (`ABP_Text_Picture_Model`)
- `utils.py`: LM embedding + metrics + graph helpers
- `features/ABPDB_protein.py`: feature extraction from PDB
- `preprocess/save_0.py`, `preprocess/save_1.py`: label ID generation from file names

## Environment Setup

1. Create and activate a Python environment (recommended Python 3.9+).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Directory Layout

Expected runtime directories:

```text
.
|-- checkpoints/
|-- data/
|   |-- ABPDB/
|      |-- seq.npy
|      |-- phipsi.npy
|      |-- DSSP.npy
|      |-- distance_value.npy
|      |-- movement_vector.npy
|      |-- quater_number.npy
|      |-- mask.npy
|      |-- label.npy
|      |-- ABPDB_7/           # train set used by train
|      |-- ABPDB_3/           # independent eval set used by train and eval script
|  
|-- features/
|-- preprocess/
|-- Rostlab/                   # local protein LM directories
`-- facebook/                  # local ESM directories
```

## Local Pretrained Language Models

The code loads models from local folders (not auto-downloaded in script). Prepare one or more of:

- `./Rostlab/prot_bert_bfd`
- `./Rostlab/prot_bert`
- `./Rostlab/prot_t5_xl_bfd`
- `./Rostlab/prot_t5_xl_uniref50`
- `./Rostlab/prot_xlnet`
- `./Rostlab/ProstT5`
- `./facebook/esm2_t6_8M_UR50D`
- `./facebook/esm2_t33_650M_UR50D`

## Data Preparation

### 1) Generate label name lists

Put PDB files into:

- `./preprocess/data_negative/`
- `./preprocess/data_positive/`

Then run:

```bash
python preprocess/save_0.py
python preprocess/save_1.py
```

This generates:

- `./features/ABPDB_label_0.npy`
- `./features/ABPDB_label_1.npy`

### 2) Extract protein features

Before running `features/ABPDB_protein.py`, adjust hard-coded variables in that file if needed:

- `sample_num`
- `Truncation_length`
- `dir_name`

Run:

```bash
python features/ABPDB_protein.py
```

It saves `.npy` files under `./data/ABPDB/`.

### 3) Split ABPDB into train/test (7:3)

Run:

```bash
python preprocess/split_abpdb.py --input_dir ./data/ABPDB/ --train_ratio 0.7 --seed 7
```

This creates:

- `./data/ABPDB/ABPDB_7/` (train)
- `./data/ABPDB/ABPDB_3/` (test)

## Input Tensor Shapes

The training/evaluation scripts expect:

- `seq.npy`: `(N, 100)`
- `phipsi.npy`: `(N, 100, 2)`
- `DSSP.npy`: `(N, 100)`
- `distance_value.npy`: `(N, 100, 100)`
- `movement_vector.npy`: `(N, 100, 100, 3)`
- `quater_number.npy`: `(N, 100, 100, 4)`
- `mask.npy`: `(N, 100)` (bool)
- `label.npy`: `(N, 1)`

## Training

Example:

```bash
python train_with_contrastive.py \
  --dir_name ./data/ABPDB/ \
  --use_lm True \
  --lm_model prot_t5_xl_uniref50
```

Resume from checkpoint:

```bash
python train_with_contrastive.py \
  --checkpoint ./checkpoints/epoch_10_use_lm_True_full.pt \
  --load_optimizer True
```

Training behavior:

- Loads all samples from `./data/ABPDB/ABPDB_7/`
- Randomly splits `ABPDB_7` into train/test with ratio `9:1` (seed controlled by `--seed`)
- If `./data/ABPDB/ABPDB_3/` exists with complete `.npy` files, it is used as an additional independent evaluation set during training

Training outputs checkpoints under `./checkpoints/`.

## Evaluation

Example:

```bash
python eval.py \
  --model_ck_filename epoch_73_use_lm_True.pt \
  --dir_name ./data/ABPDB/ABPDB_3/ \
  --use_lm True \
  --lm_model prot_t5_xl_uniref50
```

## Key Arguments

From `train_with_contrastive.py`:

- `--contrastive_weight`: contrastive loss weight (default `0.1`)
- `--distance_threshold`: neighbor filtering distance threshold (default `10`)
- `--num_epoch`: total epochs
- `--lr`: learning rate
- `--train_batch`, `--test_batch`, `--eval_batch`, `--independent_eval_batch`
- `--eval_interval`: evaluation frequency

From `eval.py`:

- `--model_ck_filename`: checkpoint filename under `./checkpoints/`
- `--dir_name`: evaluation dataset path
- `--eval_batch`: evaluation batch size

## Notes / Caveats

- Boolean CLI args use `type=bool`. Pass explicit values like `True` / `False`.
- Sequence length is effectively fixed to 100 in several places.
- `train_with_contrastive.py` uses `args.dir_name + "ABPDB_7/"` and does an internal `9:1` random split for train/test.
- `args.dir_name + "ABPDB_3/"` is optional and used only for independent evaluation during training.
- Local pretrained model folders must exist before running.
