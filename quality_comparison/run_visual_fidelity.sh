#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

REAL_ROOT="$SAVE_DIR_PATH/real_images"
SIM_ROOT="$SAVE_DIR_PATH/simulated_images"

for real_1 in "mp3d_real"
do
   for real_2 in "mp3d_real" #"fp_sim" "gibson_sim" "procthor_sim" 
   do
       echo "=========> Comparing $real_1 with $real_2"
       python measure_visual_fidelity.py \
           --real-path "$REAL_ROOT/$real_1" \
           --sim-path "$REAL_ROOT/$real_2"
   done
done


# # Comparing simulated images with real images
# for real in "mp3d_real"
# do
#    for sim in "hm3d_sim" #"fp_sim" "gibson_sim" "procthor_sim" 
#    do
#        echo "=========> Comparing $sim with $real"
#        python measure_visual_fidelity.py \
#            --real-path "$REAL_ROOT/$real" \
#            --sim-path "$SIM_ROOT/$sim"
#    done
# done


# # # Comparing real images with real images
# for sim_1 in "hm3d_sim" #"gibson_sim"
# do
#    for sim_2 in "procthor_sim" "fp_sim"
#    do
#        echo "=========> Comparing $sim_1 with $sim_2"
#        python measure_visual_fidelity.py \
#            --real-path "$SIM_ROOT/$sim_1" \
#            --sim-path "$SIM_ROOT/$sim_2"
#    done
# done

# for sim_1 in "hm3d_sim" "gibson_sim"
# do
#    for sim_2 in "hm3d_sim" "gibson_sim"
#    do
#        echo "=========> Comparing $sim_1 with $sim_2"
#        python measure_visual_fidelity.py \
#            --real-path "$SIM_ROOT/$sim_1" \
#            --sim-path "$SIM_ROOT/$sim_2"
#    done
# done
