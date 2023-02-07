#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import glob
import json
import multiprocessing as mp
import os

import numpy as np
import open3d as o3d
import pandas as pd
import tqdm
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation as R

Image.MAX_IMAGE_PIXELS = 1000000000

from typing import Any, Callable, Dict, List
import pdb

from metrics import (
    compute_floor_area,
    compute_navigable_area,
    compute_navigation_complexity,
    compute_scene_clutter,
    compute_navmesh_island_classifications,
    get_ceiling_islands,
)

from common.utils import get_filtered_scenes, robust_load_sim

os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

VALID_METRICS: List[str] = [
    "navigable_area",
    "navigation_complexity",
    "scene_clutter",
    "floor_area",
]


METRIC_TO_FN_MAP: Dict[str, Callable] = {
    "navigable_area": compute_navigable_area,
    "navigation_complexity": compute_navigation_complexity,
    "scene_clutter": compute_scene_clutter,
    "floor_area": compute_floor_area,
}


METRICS_TO_AVERAGE: List[str] = ["navigation_complexity", "scene_clutter"]

def read_json(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    return data

def get_geometry_configs(scene_instance_path, scene_dataset_cfg):
    geo_cfg = {}
    config_root = os.path.dirname(scene_dataset_cfg) + '/configs'
    scene_instance = read_json(scene_instance_path)
    stage_json = config_root + '/' + scene_instance['stage_instance']['template_name'] + '.stage_config.json'
    stage_cfg = read_json(stage_json)
    stage_geo_path = os.path.normpath(os.path.join(stage_json, '..', stage_cfg['render_asset']))
    geo_cfg[stage_geo_path] = {
        'up': stage_cfg['up'],
        'front': stage_cfg['front'],
        'translation': [0,0,0],
        'rotation': [1,0,0,0],
        'scale': [1,1,1],
    }
    object_instances = scene_instance['object_instances']
    for object_instance in object_instances:
        object_json = config_root + '/' + object_instance['template_name'] + '.object_config.json'
        object_cfg = read_json(object_json)
        object_geo_path = os.path.normpath(os.path.join(object_json, '..', object_cfg['render_asset']))
        geo_cfg[object_geo_path] = {
            'up': object_cfg['up'],
            'front': object_cfg['front'],
            'translation': object_instance['translation'],
            'rotation': object_instance['rotation'],
            'scale': object_instance['non_uniform_scale'],
        }
    return geo_cfg

def get_transformation(geo_cfg):
    front = np.asarray(geo_cfg['front'])
    front = front / np.linalg.norm(front)
    up = np.asarray(geo_cfg['up'])
    up = up / np.linalg.norm(up)
    right = np.cross(up, front)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([-right, up, -front], axis=0)
    
    transform = np.eye(4)
    translation = geo_cfg['translation']
    transform[:3, 3] = translation
    quat = geo_cfg['rotation']
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    rotation = r.as_matrix()
    transform[:3, :3] = rotation
    scale = geo_cfg['scale']
    transform[:3, :3] = transform[:3, :3].dot(np.diag(scale))
    return pose.dot(transform)

def compute_metrics(
    scene_path: str,
    scene_dataset_cfg: str,
    voxel_size: float,
    metrics: List[str] = VALID_METRICS,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Computes the 3D scene metrics for a given file.

    Args:
        scene_path: path to the scene file (glb / ply)
        voxel_size: specifies the voxel size for scene simplification
        metrics: list of metrics to compute

    Outputs:
        metric_values: a dict mapping from the required metric names to values
    """
    # sanity check
    for metric in metrics:
        assert metric in VALID_METRICS
    # load scene in habitat_simulator and trimesh
    scene_id = scene_path.split("/")[-1].replace(".scene_instance.json", "")
    hsim = robust_load_sim(scene_id, scene_dataset_cfg)
    # grabbing FP scene glbs from stage file
    with open(scene_path, "r") as f:
        scene_json = json.load(f)
    geometry_configs = get_geometry_configs(scene_path, scene_dataset_cfg)
    # scene_glb_path = os.path.join(
    #     os.path.dirname(scene_path), scene_json["render_asset"]
    # )
    triangles = None
    for geometry_path, geometry_cfg in geometry_configs.items():
        trimesh_geo = trimesh.load(geometry_path)
        tmp_triangles = trimesh_geo.triangles
        transformation = get_transformation(geometry_cfg)
        tmp_triangles = tmp_triangles.dot(transformation[:3, :3].transpose()) + transformation[:3, 3]
        if triangles is None:
            triangles = tmp_triangles
        else:
            triangles = np.concatenate((triangles, tmp_triangles), axis=0)

    # Simplify scene-mesh for faster metric computation
    # Does not impact the final metrics much
    o3d_scene = o3d.geometry.TriangleMesh()
    vertices = np.array(triangles).reshape(-1, 3)
    faces = np.arange(0, len(vertices)).reshape(-1, 3)
    o3d_scene.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_scene.triangles = o3d.utility.Vector3iVector(faces)
    o3d_scene = o3d_scene.simplify_vertex_clustering(
        voxel_size=voxel_size,
        contraction=o3d.geometry.SimplificationContraction.Average,
    )
    if verbose:
        print(
            f"=====> Downsampled mesh from {len(trimesh_scene.triangles)} "
            f"to {len(o3d_scene.triangles)}"
        )
    trimesh_scene = trimesh.Trimesh()
    trimesh_scene.vertices = np.array(o3d_scene.vertices)
    trimesh_scene.faces = np.array(o3d_scene.triangles)
    export_scenes = True
    if export_scenes:
        dataset_name = os.path.basename(os.path.dirname(scene_dataset_cfg))
        output_scene_path = os.path.join('scenes', dataset_name, f'{scene_id}.glb')
        os.makedirs(os.path.dirname(output_scene_path), exist_ok=True)
        trimesh_scene.export(output_scene_path)

    metric_values = {}
    navmesh_classification_results, indoor_islands = compute_navmesh_island_classifications(hsim)
    island_indices = np.arange(hsim.pathfinder.num_islands)
    outdoor_islands = np.setdiff1d(island_indices, indoor_islands)
    if len(outdoor_islands):
        ceiling_islands = get_ceiling_islands(hsim, outdoor_islands, trimesh_scene)
        # remove ceiling islands
        outdoor_islands = np.setdiff1d(outdoor_islands, ceiling_islands)
    for metric in metrics:
        metric_values[metric] = METRIC_TO_FN_MAP[metric](hsim, trimesh_scene, scene_id, indoor_islands=indoor_islands, outdoor_islands=outdoor_islands, navigable_area=metric_values.get('navigable_area'))
    metric_values["scene"] = scene_path.split("/")[-1].split(".")[0]
    hsim.close()
    return metric_values


def _aux_fn(inputs: Any) -> Any:
    print('########################################################\n')
    print(f'scene{inputs[0]}\n')
    try:
        results = compute_metrics(*inputs)
    except Exception as e:
        results = {"scene": inputs[0]}
        raise e
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root", type=str, required=True, help="path to FP stages directory"
    )
    parser.add_argument("--metrics", type=str, nargs="+", default=VALID_METRICS)
    parser.add_argument("--filter-scenes", type=str, default="")
    parser.add_argument("--scene-id", type=str, default="")
    parser.add_argument("--save-path", type=str, default="")
    parser.add_argument("--scene-dataset-cfg", type=str, required=True)
    parser.add_argument("--scan-patterns", type=str, nargs="+", default=["**/*.scene_instance.json"])
    parser.add_argument("--voxel-size", type=float, default=0.1)
    parser.add_argument("--n-processes", type=int, default=4)
    parser.add_argument("--verbose", action="store_true", default=False)

    args = parser.parse_args()

    scenes = []
    for scan_pattern in args.scan_patterns:
        scenes += glob.glob(f"{args.dataset_root}/{scan_pattern}", recursive=True)
    if args.filter_scenes != "":
        scenes = get_filtered_scenes(scenes, args.filter_scenes, args.scene_id)
    scenes = sorted(scenes)
    # Filter out basis scenes
    scenes = [s for s in scenes if ".basis." not in s]

    if args.verbose:
        print(f"Number of scenes in {args.dataset_root}: {len(scenes)}")

    context = mp.get_context("forkserver")
    inputs = [
        [scene, args.scene_dataset_cfg, args.voxel_size, args.metrics, args.verbose]
        for scene in scenes
    ]

    if args.n_processes == 1:
        stats = []
        num_scenes = len(scenes)
        for i in tqdm.tqdm(range(num_scenes)):
            stats.append(_aux_fn(inputs[i]))
    else:
        pool = context.Pool(processes=args.n_processes, maxtasksperchild=2)
        stats = list(tqdm.tqdm(pool.imap(_aux_fn, inputs), total=len(scenes)))
    
    stats = pd.DataFrame(stats)
    stats.set_index("scene", inplace=True)
    for column in stats.columns:
        stats = pd.concat([stats, stats[column].apply(pd.Series)], axis=1)
        stats.drop(columns=column, inplace=True)
    print("============= Metrics =============")
    print(f"Number of scenes: {len(scenes)}")
    for metric in args.metrics:
        if metric in METRICS_TO_AVERAGE:
            indoor_v = stats[f'indoor_{metric}'].to_numpy().mean().item()
            outdoor_v = stats[f'outdoor_{metric}'].to_numpy().mean().item()
            total_v = stats[f'total_{metric}'].to_numpy().mean().item()
        else:
            indoor_v = stats[f'indoor_{metric}'].to_numpy().sum().item()
            outdoor_v = stats[f'outdoor_{metric}'].to_numpy().sum().item()
            total_v = stats[f'total_{metric}'].to_numpy().sum().item()
        print(f"indoor_{metric:<30s} | {indoor_v:.3f}")
        print(f"outdoor_{metric:<30s} | {outdoor_v:.3f}")
        print(f"total_{metric:<30s} | {total_v:.3f}")

    if args.save_path != "":
        if args.scene_id:
            output_path = os.path.splitext(args.save_path)[0] + f'_{args.scene_id}' + os.path.splitext(args.save_path)[-1]
            stats.to_csv(output_path, sep="\t")
        else:
            stats.to_csv(args.save_path, sep="\t")
