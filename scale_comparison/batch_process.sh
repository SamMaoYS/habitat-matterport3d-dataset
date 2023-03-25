#!/bin/bash

# SCRIPT=compute_scene_metrics.py
# DATASET_ROOT=data/scene_datasets/fphab-v0.2.0
# DATASET_CFG=hab-fp.scene_dataset_config.json
# OUTPUT_PATH=data/fp_metrics/fp_metrics.csv
# SCENE_IDS=data/scene_datasets/fphab.txt
# PARALLEL_N=1

SCRIPT=compute_scene_metrics.py
DATASET_ROOT=data/scene_datasets/ai2thor-hab-v0.0.9
DATASET_CFG=ai2thor.scene_dataset_config.json
OUTPUT_PATH=data/ithor_metrics_fa/ithor_metrics_fa.csv
SCENE_IDS=data/scene_datasets/iTHOR.txt
PARALLEL_N=4

echo "===================================="
echo "Processing dataset ..."
echo "===================================="

parallel -j $PARALLEL_N --bar "python $SCRIPT \
    --dataset-root $DATASET_ROOT \
    --scene-dataset-cfg $DATASET_ROOT/$DATASET_CFG \
    --filter-scenes $SCENE_IDS \
    --n-processes 1 \
    --save-path $OUTPUT_PATH \
    --scene-id {1}" :::: $SCENE_IDS