# FACL-PropPred

> Official implementation of **FACL**, a fragment-aware contrastive learning framework for molecular property prediction.

<p align="center">
  <img src="https://github.com/user-attachments/assets/afec4778-ebfa-49f0-bd74-05542e2a7ffe" alt="Overview of the FACL framework" width="90%">
</p>

## Overview

FACL is a molecular representation learning framework designed for molecular property prediction. It incorporates fragment-level information into contrastive learning and supports the complete workflow from fragment database construction and contrastive-pair precomputation to pre-training and downstream fine-tuning.

## Table of Contents

- [Installation](#installation)
- [Data](#data)
- [FGIB Training](#fgib-training)
- [Fragment Database Preparation](#fragment-database-preparation)
- [Contrastive-Pair Precomputation](#contrastive-pair-precomputation)
- [Pre-training](#pre-training)
- [Fine-tuning](#fine-tuning)
- [Pre-trained Models](#pre-trained-models)

## Installation

Clone the repository and create the Conda environment:

```bash
git clone https://github.com/XM001-hub/FACL-propred.git
cd FACL-propred
conda env create -f environment.yml
```

Activate the environment after installation:

```bash
conda activate <environment-name>
```

> Replace `<environment-name>` with the environment name defined in `environment.yml`.

## Data

The raw datasets used for pre-training and fine-tuning are located in the [`dataset`](./dataset) directory.

## FGIB Training

To train the FGIB module, run:

```bash
cd torch_fgib
python train_fgib.py
```

Return to the project root before running the remaining commands:

```bash
cd ..
```

## Fragment Database Preparation

You can build a custom fragment database for constructing contrastive pairs. The following example generates fragments with the context radius set to `0`:

```bash
python create_frag_env_db.py \
  -smiles_path /path/to/smiles.csv \
  -db_path /path/to/core_fragments.db \
  -radius 0
```

### Arguments

| Argument | Description |
| --- | --- |
| `-smiles_path` | Path to the input CSV file containing SMILES strings |
| `-db_path` | Path where the generated fragment database will be saved |
| `-radius` | Radius of the fragment context |

## Contrastive-Pair Precomputation

After generating the fragment database, precompute molecular contrastive pairs for the ZINC dataset. This step can significantly accelerate pre-training:

```bash
python precompute.py \
  -smiles_path /path/to/smiles.csv \
  -fragment_path /path/to/core_fragments.db
```

## Pre-training

Pre-training settings—including the model backbone and number of layers—can be configured in [`config/facl_config.yml`](./config/facl_config.yml).

Once the configuration is ready, start pre-training with:

```bash
python pretrain.py
```

The resulting pre-trained model will be saved to the output directory configured by the project.

## Fine-tuning

Two fine-tuning strategies are available for downstream molecular property prediction.

### Standard fine-tuning

Fine-tune the pre-trained model without updating the FGIB module:

```bash
python finetune.py
```

### Joint fine-tuning with FGIB

Fine-tune the pre-trained model together with the FGIB module:

```bash
python dual_finetune.py
```

Joint fine-tuning typically provides better prediction performance.

## Pre-trained Models

Pre-trained model checkpoints are available from [Google Drive](https://drive.google.com/drive/folders/1n9RFJVajxdUglznCqaYT9s1cOp2IHlOg?usp=sharing).

## Citation

If you find this repository useful, please consider citing the associated work:

> *Fragment-aware contrastive learning framework for molecular property prediction*

The complete citation and BibTeX entry can be added here when they become available.

## Acknowledgements

Thank you for your interest in FACL-PropPred. Questions, bug reports, and contributions are welcome through GitHub Issues and Pull Requests.


