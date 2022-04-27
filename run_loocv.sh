#!/bin/bash
#SBATCH --partition=bigmem
#SBATCH --job-name=run_loocv
#SBATCH --account=nal_genomics
#SBATCH --mail-user=<usermail>
#SBATCH --mail-type=NONE
#SBATCH --time=40:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=48


export SAVE_DIR=./output
export MAX_LENGTH=50
export BATCH_SIZE=32
export NUM_EPOCHS=8
export SAVE_STEPS=1000
export SEED=1

for SPLIT in {0..192}
do
    DATA_DIR=./LOOCV_dataset/${SPLIT}
    ENTITY=${SPLIT}
    echo "***** " $DATA " train-eval " $SPLIT " Start *****"
    python run_ner.py \
    --data_dir ${DATA_DIR}/ \
    --model_name_or_path ./biobert-base-cased-v1.1 \
    --output_dir ${SAVE_DIR}/${ENTITY} \
    --labels ${DATA_DIR}/labels.txt \
    --max_seq_length ${MAX_LENGTH} \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --save_steps ${SAVE_STEPS} \
    --seed ${SEED} \
    --do_train \
    --do_eval \
    --do_predict \
    --overwrite_output_dir

done
echo "***** " $SPLIT " train-eval Done *****"
