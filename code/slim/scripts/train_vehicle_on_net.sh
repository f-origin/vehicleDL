#!/bin/bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script performs the following operations:
# 1. Downloads the Cifar10 dataset
# 2. Trains a CifarNet model on the Cifar10 training set.
# 3. Evaluates the model on the Cifar10 testing set.
#
# Usage:
# cd slim
# ./scripts/train_cifarnet_on_cifar10.sh
set -e

echo "run train vehicle shell"
# Where the checkpoint and logs will be saved to.
OUT_DIR=/output
# OUT_DIR=~/tmp

TRAIN_DIR=${OUT_DIR}/vehicle-model

# Where the dataset is saved to.
DATASET_DIR=/data/forigin/car-detction
# DATASET_DIR=~/tmp/vehicle

DATASET_NAME=pj_vehicle

# Model name
MODEL_NAME=inception_v4

# Run training.
python3 train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME} \
  --checkpoint_path=${DATASET_DIR}/inception_v4.ckpt \
  --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
  --preprocessing_name=inception \
  --max_number_of_steps=500 \
  --batch_size=12 \
  --save_interval_secs=120 \
  --save_summaries_secs=120 \
  --log_every_n_steps=100 \
  --optimizer=adam \
  --learning_rate=1 \
  --num_epochs_per_decay=200 \
  --weight_decay=0.004

# Run evaluation.
python3 eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME}

# Run export.
python3 export_inference_graph.py \
  --model_name=${MODEL_NAME} \
  --batch_size=1 \
  --dataset_name=${DATASET_NAME} \
  --dataset_dir=${DATASET_DIR} \
  --output_file=${TRAIN_DIR}/inception_v3_inf_graph.pb

#cd ${OUT_DIR}
#tar -zvcf model_exported.tar.gz vehicle-model/
