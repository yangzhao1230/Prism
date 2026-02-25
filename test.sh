#!/bin/bash
# Run inference on all cell types and seeds using pre-trained checkpoints.
#
# Usage: bash test.sh /path/to/data /path/to/ckpt
#
# Data: https://huggingface.co/datasets/xingyusu/GeneExp
# Checkpoints: https://huggingface.co/yangyz1230/Prism
#
# Download checkpoints:
#   python -c "from huggingface_hub import snapshot_download; snapshot_download('yangyz1230/Prism', repo_type='model', local_dir='./ckpt')"

set -e

SEQLEN=2000
BATCH_SIZE=8
DATA_ROOT=$(realpath "${1:?Usage: bash test.sh /path/to/data /path/to/ckpt}")
CKPT_BASE_DIR=$(realpath "${2:?Usage: bash test.sh /path/to/data /path/to/ckpt}")
Cell_Types=(K562 GM12878)
N_CONTEXT=2
CNN_DIM=8
UNIFORM_LOSS_WEIGHT=1.0
INTERVENTION_LOSS_WEIGHT=1.0
DEVICES=${DEVICES:-1}
WANDB_MODE=${WANDB_MODE:-disabled}
SEEDS=(2 22 222 2222 22222)

for Cell_Type in "${Cell_Types[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        CKPT_DIR="${CKPT_BASE_DIR}/${Cell_Type}/seed_${SEED}"
        MODEL_PATH=$(ls "${CKPT_DIR}"/*.ckpt 2>/dev/null | sort | tail -1)
        if [ -z "${MODEL_PATH}" ]; then
            echo "WARNING: No checkpoint found in ${CKPT_DIR}, skipping ${Cell_Type} seed ${SEED}"
            continue
        fi
        echo "Running inference: Cell_Type=${Cell_Type}, SEED=${SEED}"
        echo "  checkpoint: ${MODEL_PATH}"

        run_name="test_${Cell_Type}_seed_${SEED}"

        python -m train \
            experiment=hg38/gene_express \
            wandb.mode=${WANDB_MODE} \
            wandb.group="CAGE_${Cell_Type}" \
            wandb.name=${run_name} \
            hydra.run.dir="./outputs/prism_test/${Cell_Type}_seed_${SEED}" \
            dataset.expr_type=CAGE \
            dataset.cell_type="$Cell_Type" \
            model="prism" \
            dataset.batch_size=$((BATCH_SIZE / DEVICES)) \
            trainer.devices=${DEVICES} \
            model.config.n_layer=4 \
            model.config.n_context=${N_CONTEXT} \
            model.config.cnn_dim=${CNN_DIM} \
            dataset.data_folder="$DATA_ROOT" \
            dataset.seq_range=${SEQLEN} \
            train.remove_test_loader_in_eval=False \
            task="extract_rationale" \
            task.loss.uniform_loss_weight=${UNIFORM_LOSS_WEIGHT} \
            task.loss.intervention_loss_weight=${INTERVENTION_LOSS_WEIGHT} \
            model.config.intervention_loss_weight=${INTERVENTION_LOSS_WEIGHT} \
            train.seed=${SEED} \
            wandb.project=prism \
            train.only_test=True \
            train.only_test_model_path="${MODEL_PATH}"
    done
done
