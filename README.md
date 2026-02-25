# Prism

Official implementation of the paper:

> **Extending Sequence Length is Not All You Need: Effective Integration of Multimodal Signals for Gene Expression Prediction** (ICLR 2026)

## Installation

Requires Python 3.9+, CUDA 12.x, and a compatible GPU.

```bash
conda create -n prism python=3.9 -y
conda activate prism
pip install -r requirements.txt
```

## Data

We use the same dataset as [Seq2Exp](https://github.com/divelab/AIRS/tree/main/OpenBio/Seq2Exp). Download from HuggingFace:

```bash
pip install huggingface_hub

python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='xingyusu/GeneExp',
    repo_type='dataset',
    local_dir='./data'
)
"
```

Set the data directory:

```bash
DATA_ROOT=/path/to/data
```

## Model Weights

Pre-trained checkpoints for both K562 and GM12878 (5 seeds each) are available on HuggingFace:

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='yangyz1230/Prism',
    repo_type='model',
    local_dir='./ckpt'
)
"
```

This gives you the following structure:

```
ckpt/
├── K562/
│   ├── seed_2/mse-2025-05-13_19-53-15+Fold11.ckpt
│   ├── seed_22/...
│   ├── seed_222/...
│   ├── seed_2222/...
│   └── seed_22222/...
└── GM12878/
    ├── seed_2/mse-2025-05-14_01-01-52+Fold11.ckpt
    ├── seed_22/...
    ├── seed_222/...
    ├── seed_2222/...
    └── seed_22222/...
```

## Inference

Run inference on all cell types and seeds using the provided checkpoints:

```bash
bash test.sh $DATA_ROOT ./ckpt
```

The script auto-discovers the best checkpoint for each `(cell_type, seed)` combination. Results are printed to stdout. By default, inference runs on 1 GPU with W&B logging disabled.

## Training

Train Prism from scratch on K562 with a single seed:

```bash
bash train_example.sh
```

To train on multiple seeds or cell types, modify the variables in `train_example.sh`.

## Code Structure

```
.
├── train.py                          # Main entry point
├── train_example.sh                  # Training script
├── test.sh                           # Inference script
├── requirements.txt                  # Python dependencies
├── configs/                          # Hydra configuration files
│   ├── config.yaml
│   ├── experiment/hg38/gene_express.yaml
│   └── model/prism.yaml
├── src/
│   ├── models/sequence/
│   │   └── GeneExpformer.py          # Prism model & SignalWeightGenerator
│   ├── tasks/
│   │   └── metrics.py                # Loss functions (L1, L2, L3)
│   └── dataloaders/
│       └── datasets/
│           └── promo_enhan_inter.py  # Dataset & signal loading
└── caduceus/                         # Caduceus backbone (bidirectional Mamba)
```

## Citation

```bibtex
@inproceedings{
yang2026extending,
title={Extending Sequence Length is Not All You Need: Effective Integration of Multimodal Signals for Gene Expression Prediction},
author={Zhao Yang and Yi Duan and Jiwei Zhu and Ying Ba and Chuan Cao and Bing Su},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026}
}
```

## Acknowledgements

Our codebase is built upon [Seq2Exp](https://github.com/divelab/AIRS/tree/main/OpenBio/Seq2Exp). We thank the authors for making their implementation publicly available.
