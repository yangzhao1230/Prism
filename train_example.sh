#!/bin/bash
# Example: train Prism on K562 with a single seed.
# Usage: bash train_example.sh /path/to/data
#
# Data can be downloaded from HuggingFace:
#   python -c "from huggingface_hub import snapshot_download; snapshot_download('xingyusu/GeneExp', repo_type='dataset', local_dir='./data')"

set -e

export WANDB_API_KEY="YOUR_WANDB_API_KEY"

SEQLEN=2000
BATCH_SIZE=8
DATA_ROOT=${1:?Usage: bash train_example.sh /path/to/data}
Cell_Type=K562
N_CONTEXT=2
CNN_DIM=8
UNIFORM_LOSS_WEIGHT=1.0
INTERVENTION_LOSS_WEIGHT=1.0
SEED=2
DEVICES=1
WANDB_MODE=${WANDB_MODE:-disabled}

run_name="prism_${Cell_Type}_n_context_${N_CONTEXT}_cnn_dim_${CNN_DIM}_len_${SEQLEN}_u_weight_${UNIFORM_LOSS_WEIGHT}_i_weight_${INTERVENTION_LOSS_WEIGHT}_seed_${SEED}"

python -m train \
    experiment=hg38/gene_express \
    wandb.mode=${WANDB_MODE} \
    wandb.group="CAGE_${Cell_Type}" \
    wandb.name=${run_name} \
    hydra.run.dir="./outputs/prism/${run_name}" \
    dataset.expr_type=CAGE \
    dataset.cell_type="${Cell_Type}" \
    model="prism" \
    dataset.batch_size=$((BATCH_SIZE / DEVICES)) \
    trainer.devices=${DEVICES} \
    model.config.n_layer=4 \
    model.config.n_context=${N_CONTEXT} \
    model.config.cnn_dim=${CNN_DIM} \
    dataset.data_folder="${DATA_ROOT}" \
    dataset.seq_range=${SEQLEN} \
    train.remove_test_loader_in_eval=True \
    task="extract_rationale" \
    task.loss.uniform_loss_weight=${UNIFORM_LOSS_WEIGHT} \
    task.loss.intervention_loss_weight=${INTERVENTION_LOSS_WEIGHT} \
    model.config.intervention_loss_weight=${INTERVENTION_LOSS_WEIGHT} \
    train.seed=${SEED} \
    wandb.project=prism
