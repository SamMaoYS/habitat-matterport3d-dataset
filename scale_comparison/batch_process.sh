#!/bin/bash

SCRIPT=compute_scene_metrics.py
DATASET_ROOT=data/scene_datasets/hssd-hab
DATASET_CFG=hssd-hab.scene_dataset_config.json
OUTPUT_PATH=data/hssd_metrics/hssd_metrics.csv
SCENE_IDS=data/scene_datasets/hssd.txt
PARALLEL_N=8

# SCRIPT=compute_scene_metrics.py
# DATASET_ROOT=data/scene_datasets/ai2thor-hab-v0.0.9
# DATASET_CFG=ai2thor.scene_dataset_config.json
# OUTPUT_PATH=data/procthor_metrics/procthor_metrics.csv
# SCENE_IDS=data/scene_datasets/ProcTHOR.txt
# PARALLEL_N=8

# SCRIPT=compute_scene_metrics.py
# DATASET_ROOT=/localhome/yma50/Development/proj-scene-builder/habitat-matterport3d-dataset/dataset/mp3d
# DATASET_CFG=mp3d.scene_dataset_config.json
# OUTPUT_PATH=data/mp3d_metrics/mp3d_metrics.csv
# SCENE_IDS=data/scene_datasets/mp3d.txt
# PARALLEL_N=16

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
