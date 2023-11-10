#!/bin/bash

# SCRIPT=compute_scene_metrics.py
# DATASET_ROOT=data/scene_datasets/fphab-v0.2.0
# DATASET_CFG=hab-fp.scene_dataset_config.json
# OUTPUT_PATH=data/fp_metrics/fp_metrics.csv
# SCENE_IDS=data/scene_datasets/fphab.txt
# PARALLEL_N=1

SCRIPT=compute_scene_metrics.py
DATASET_ROOT=data/scene_datasets/gibson_tiny
OUTPUT_PATH=data/gibson_tiny_metrics/gibson_tiny_metrics.csv
SCENE_IDS=data/scene_datasets/gibson_tiny.txt
PARALLEL_N=32

echo "===================================="
echo "Processing dataset ..."
echo "===================================="

parallel -j $PARALLEL_N --bar "python $SCRIPT \
    --dataset-root $DATASET_ROOT \
    --filter-scenes $SCENE_IDS \
    --n-processes 1 \
    --save-path $OUTPUT_PATH \
    --scene-id {1}" :::: $SCENE_IDS
