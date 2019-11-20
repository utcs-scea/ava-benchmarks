#!/bin/bash

export DATA_DIR=gs://cloud-tpu-test-datasets/fake_imagenet
export STORAGE_BUCKET=gs://tpu-vm-test

python inception_v3_mod2.py \
    --tpu=$TPU_NAME \
    --learning_rate=0.165 \
    --train_steps=10 \
    --iterations=10 \
    --use_tpu=True \
    --use_data=real \
    --mode=train_and_eval \
    --train_steps_per_eval=10 \
    --data_dir=${DATA_DIR} \
    --model_dir=${STORAGE_BUCKET}/inception
