#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

REAL_ROOT="$SAVE_DIR_PATH/simulated_images"
SIM_ROOT="$SAVE_DIR_PATH/simulated_images"


Comparing simulated images with real images
for real in "hm3d_sim"
do
   for sim in "fp_sim" "procthor_sim"
   do
       echo "=========> Comparing $sim with $real"
       python measure_visual_fidelity.py \
           --real-path "$REAL_ROOT/$real" \
           --sim-path "$SIM_ROOT/$sim"
   done
done


# Comparing real images with real images
for real_1 in "hm3d_sim"
do
   for real_2 in "hm3d_sim"
   do
       echo "=========> Comparing $real_1 with $real_2"
       python measure_visual_fidelity.py \
           --real-path "$REAL_ROOT/$real_1" \
           --sim-path "$REAL_ROOT/$real_2"
   done
done
