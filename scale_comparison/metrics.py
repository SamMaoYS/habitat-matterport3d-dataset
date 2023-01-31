#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
from typing import Any, Dict, List

import magnum as mn
import habitat_sim
import numpy as np
import scipy
import trimesh
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
import pdb
EPS = 1e-10

def island_indoor_metric(
        hsim: habitat_sim.Simulator, island_ix: int, num_samples=100, jitter_dist=0.1, max_tries=1000
    ) -> float:
    """
    Compute a heuristic for ratio of an island inside vs. outside based on checking whether there is a roof over a set of sampled navmesh points.
    """

    assert hsim.pathfinder.is_loaded
    assert hsim.pathfinder.num_islands > island_ix

    # collect jittered samples
    samples = []
    for _sample_ix in range(max_tries):
        new_sample = hsim.pathfinder.get_random_navigable_point(
            island_index=island_ix
        )
        too_close = False
        for prev_sample in samples:
            dist_to = np.linalg.norm(prev_sample - new_sample)
            if dist_to < jitter_dist:
                too_close = True
                break
        if not too_close:
            samples.append(new_sample)
        if len(samples) >= num_samples:
            break

    # classify samples
    indoor_count = 0
    for sample in samples:
        raycast_results = hsim.cast_ray(
            habitat_sim.geo.Ray(sample, mn.Vector3(0, 1, 0))
        )
        if raycast_results.has_hits():
            # assume any hit indicates "indoor"
            indoor_count += 1

    # return the ration of indoor to outdoor as the metric
    return indoor_count / len(samples)

def compute_navmesh_island_classifications(hsim: habitat_sim.Simulator, active_indoor_threshold=0.85):
    """
    Classify navmeshes as outdoor or indoor and find the largest indoor island.
    active_indoor_threshold is acceptacle indoor|outdoor ration for an active island (for example to allow some islands with a small porch or skylight)
    """
    if not hsim.pathfinder.is_loaded:
        navmesh_classification_results = None
        print("No NavMesh loaded to visualize.")
        return

    navmesh_classification_results = {}

    navmesh_classification_results["active_island"] = -1
    navmesh_classification_results[
        "active_indoor_threshold"
    ] = active_indoor_threshold
    active_island_size = 0
    number_of_indoor = 0
    navmesh_classification_results["island_info"] = {}
    indoor_islands = []

    for island_ix in range(hsim.pathfinder.num_islands):
        navmesh_classification_results["island_info"][island_ix] = {}
        navmesh_classification_results["island_info"][island_ix][
            "indoor"
        ] = island_indoor_metric(hsim, island_ix=island_ix)
        if (
            navmesh_classification_results["island_info"][island_ix]["indoor"]
            > active_indoor_threshold
        ):
            number_of_indoor += 1
            indoor_islands.append(island_ix)
        island_size = hsim.pathfinder.island_area(island_ix)

        if (
            active_island_size < island_size
            and navmesh_classification_results["island_info"][island_ix][
                "indoor"
            ]
            > active_indoor_threshold
        ):
            active_island_size = island_size
            navmesh_classification_results["active_island"] = island_ix
    # print(
    #     f"Found active island {navmesh_classification_results['active_island']} with area {active_island_size}."
    # )
    # print(
    #     f"     Found {number_of_indoor} indoor islands out of {hsim.pathfinder.num_islands} total."
    # )
    for island_ix in range(hsim.pathfinder.num_islands):
        island_info = navmesh_classification_results["island_info"][island_ix]
        info_str = f"    {island_ix}: indoor ratio = {island_info['indoor']}, area = {hsim.pathfinder.island_area(island_ix)}"
        if navmesh_classification_results["active_island"] == island_ix:
            info_str += "  -- active--"
        # print(info_str)
    return navmesh_classification_results, indoor_islands


def get_geodesic_distance(
    hsim: habitat_sim.Simulator, p1: np.ndarray, p2: np.ndarray
) -> float:
    """Computes the geodesic distance between two points."""
    path = habitat_sim.ShortestPath()
    path.requested_start = p1
    path.requested_end = p2
    hsim.pathfinder.find_path(path)
    return path.geodesic_distance


def get_euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Computes the euclidean distance between two points."""
    return np.linalg.norm(p1 - p2).item()


def get_navcomplexity(
    hsim: habitat_sim.Simulator, p1: np.ndarray, p2: np.ndarray
) -> float:
    """Computes the navigation complexity between two points in a scene."""
    geod = get_geodesic_distance(hsim, p1, p2)
    eucd = get_euclidean_distance(p1, p2)
    return geod / (eucd + EPS)


def get_triangle_areas(triangles: np.ndarray) -> np.ndarray:
    """
    Measure the area of mesh triangles.
    Args:
        triangles: (N, 3, 3) ndarray with dimension 1 representing 3 vertices
    """
    vtr10 = triangles[:, 1] - triangles[:, 0]  # (N, 3)
    vtr20 = triangles[:, 2] - triangles[:, 0]  # (N, 3)
    area = 0.5 * np.linalg.norm(np.abs(np.cross(vtr10, vtr20, axis=1)), axis=1)
    return area


def transform_coordinates_hsim_to_trimesh(xyz: np.ndarray) -> np.ndarray:
    """
    Transforms points from hsim coordinates to trimesh.

    Habitat conventions: X is rightward, Y is upward, -Z is forward
    Trimesh conventions: X is rightward, Y is forward, Z is upward

    Args:
        xyz: (N, 3) array of coordinates
    """
    xyz_trimesh = np.stack([xyz[:, 0], -xyz[:, 2], xyz[:, 1]], axis=1)
    return xyz_trimesh


def get_ceiling_islands(
    hsim: habitat_sim.Simulator,
    outdoor_islands: list,
    trimesh_scene: trimesh.parent.Geometry,
    floor_limit: float = 0.5,
    ceiling_threshold: float = 0.01,
) -> list:
    vertices = trimesh_scene.vertices
    outdoor_floor_extents = get_floor_navigable_extents(hsim, islands=outdoor_islands, num_points_to_sample=2000)
    max_height = 0
    for fext in outdoor_floor_extents:
        if max_height < fext["mean"]:
            max_height = fext["mean"]
    
    ceiling_islands = []
    for island_index in outdoor_islands:
        navmesh_vertices = np.array(hsim.pathfinder.build_navmesh_vertices(island_index))
        mean_height = np.mean(navmesh_vertices[:, 1], axis=0)
        num_verts_above = np.sum(vertices[:, 1] > (mean_height + floor_limit))
        percent_verts_above = num_verts_above / len(vertices)
        if max_height - floor_limit < mean_height and percent_verts_above < ceiling_threshold:
            ceiling_islands.append(island_index)
    return ceiling_islands

def get_floor_navigable_extents(
    hsim: habitat_sim.Simulator, *args: Any, num_points_to_sample: int = 20000, **kwargs: Any
) -> List[Dict[str, float]]:
    """
    Function to estimate the number of floors in a 3D scene and the Y extents
    of the navigable space on each floor. It samples a random number
    of navigable points and clusters them based on their height coordinate.
    Each cluster corresponds to a floor, and the points within a cluster
    determine the extents of the navigable surfaces in a floor.
    """
    # randomly sample navigable points
    random_navigable_points = []
    islands = kwargs['islands']
    for island_index in islands:
        tmp_random_navigable_points= []
        for _i in range(num_points_to_sample):
            point = hsim.pathfinder.get_random_navigable_point(island_index=island_index)
            if np.isnan(point).any() or np.isinf(point).any():
                continue
            tmp_random_navigable_points.append(point)
        random_navigable_points += tmp_random_navigable_points
    random_navigable_points = np.array(random_navigable_points)
    # cluster the rounded y_coordinates using DBScan
    y_coors = np.around(random_navigable_points[:, 1], decimals=1)
    clustering = DBSCAN(eps=0.2, min_samples=500).fit(y_coors[:, np.newaxis])
    c_labels = clustering.labels_
    n_clusters = len(set(c_labels)) - (1 if -1 in c_labels else 0)
    # estimate floor extents
    floor_extents = []
    core_sample_y = y_coors[clustering.core_sample_indices_]
    core_sample_labels = c_labels[clustering.core_sample_indices_]
    for i in range(n_clusters):
        floor_min = core_sample_y[core_sample_labels == i].min().item()
        floor_max = core_sample_y[core_sample_labels == i].max().item()
        floor_mean = core_sample_y[core_sample_labels == i].mean().item()
        floor_extents.append({"min": floor_min, "max": floor_max, "mean": floor_mean})
    return floor_extents


def compute_navigable_area(
    hsim: habitat_sim.Simulator, *args: Any, **kwargs: Any
) -> dict:
    """
    Navigable area (m^2) measures the total scene area that is actually
    navigable in the scene. This is computed for a cylindrical robot with radius
    0.1m and height 1.5m using the AI Habitat navigation mesh implementation.
    This excludes points that are not reachable by the robot. Higher values
    indicate larger quantity and diversity of viewpoints for a robot.
    """
    indoor_islands = kwargs['indoor_islands']
    outdoor_islands = kwargs['outdoor_islands']
    indoor_navigable_area = np.sum([hsim.pathfinder.island_area(island) for island in indoor_islands])
    outdoor_navigable_area = np.sum([hsim.pathfinder.island_area(island) for island in outdoor_islands])

    navigable_area = {
        'indoor_navigable_area': indoor_navigable_area,
        'outdoor_navigable_area': outdoor_navigable_area,
        'total_navigable_area': hsim.pathfinder.navigable_area
    }
    return navigable_area

def compute_navigation_complexity_impl(
    hsim: habitat_sim.Simulator,
    islands: list,
    max_pairs_to_sample: int = 20000,
    max_trials_per_pair: int = 10,
) -> float:
    if not hsim.pathfinder.is_loaded:
        return 0.0
    navcomplexity = 0.0
    num_sampled_pairs = 0
    for island_index in islands:
        while num_sampled_pairs < max_pairs_to_sample:
            num_sampled_pairs += 1
            p1 = hsim.pathfinder.get_random_navigable_point(island_index=island_index)
            num_trials = 0
            while num_trials < max_trials_per_pair:
                num_trials += 1
                p2 = hsim.pathfinder.get_random_navigable_point(island_index=island_index)
                # Different floors
                if abs(p1[1] - p2[1]) > 0.5:
                    continue
                cur_navcomplexity = get_navcomplexity(hsim, p1, p2)
                # Ignore disconnected pairs
                if math.isinf(cur_navcomplexity):
                    continue
                navcomplexity = max(navcomplexity, cur_navcomplexity)
    return navcomplexity

def compute_navigation_complexity(
    hsim: habitat_sim.Simulator,
    *args: Any,
    max_pairs_to_sample: int = 5000,
    max_trials_per_pair: int = 10,
    **kwargs: Any,
) -> float:
    """
    Navigation complexity measures the difficulty of navigating in a scene.
    This is computed as the maximum ratio of geodesic path to euclidean
    distances between any two navigable locations in the scene. Higher values
    indicate more complex layouts with navigation paths that deviate
    significantly from straight-line paths.

    Args:
        hsim: habitat simulator instance
        max_pairs_to_sample: the maximum number of random point pairs to sample
        max_trials_per_pair: the maximum trials to find a paired point p2 for
            a given point p1
    """
    if not hsim.pathfinder.is_loaded:
        return 0.0
    indoor_islands = kwargs['indoor_islands']
    outdoor_islands = kwargs['outdoor_islands']
    indoor_navigation_complexity = compute_navigation_complexity_impl(hsim, indoor_islands, max_pairs_to_sample, max_trials_per_pair)
    outdoor_navigation_complexity = compute_navigation_complexity_impl(hsim, outdoor_islands, max_pairs_to_sample, max_trials_per_pair)
    total_navigation_complexity = compute_navigation_complexity_impl(hsim, [-1])
    navcomplexity = {
        'indoor_navigation_complexity': indoor_navigation_complexity,
        'outdoor_navigation_complexity': outdoor_navigation_complexity,
        'total_navigation_complexity': total_navigation_complexity
    }
    return navcomplexity

def compute_scene_clutter_impl(
    trimesh_scene: trimesh.parent.Geometry,
    navmesh_vertices: np.ndarray,
    navmesh_area: float,
    *args: Any,
    closeness_thresh: float = 0.5,
    **kwargs: Any,
) -> float:
    mesh_triangles = np.copy(trimesh_scene.triangles)
    navmesh_faces = np.arange(0, navmesh_vertices.shape[0], dtype=np.uint32)
    navmesh_faces = navmesh_faces.reshape(-1, 3)
    navmesh_triangles = navmesh_vertices.reshape(-1, 3, 3)
    navmesh_centroids = navmesh_triangles.mean(axis=1)
    navmesh = trimesh.Trimesh(vertices=navmesh_vertices, faces=navmesh_faces)

    # visualization
    visualization = True
    if visualization:
        navmesh_id = kwargs['navmesh_id']
        from visualization import Visualizer
        viz = Visualizer()
        scene_pcd = trimesh.PointCloud(vertices=trimesh_scene.vertices, colors=[0, 255, 0])
        viz.add_geometry(scene_pcd)
        viz.add_geometry(navmesh)
        color, _ = viz.render()
        from PIL import Image
        img = Image.fromarray(color.astype('uint8'), 'RGBA')
        os.makedirs('navmesh-render', exist_ok=True)
        img.save(f'navmesh-render/{navmesh_id}.png')
        navmesh.export(f'navmesh-render/{navmesh_id}.ply')

    # Find closest distance between a mesh_triangle and the navmesh
    # This is approximated by measuring the distance between each vertex and
    # centroid of a mesh_triangle to the navmesh surface
    ## (1) pre-filtering to remove unrelated mesh_triangles
    tree = scipy.spatial.cKDTree(navmesh_centroids)
    mesh_centroids = mesh_triangles.mean(axis=1)[:, np.newaxis, :]
    mindist, _ = tree.query(mesh_centroids)
    valid_mask = mindist[:, 0] <= 2 * closeness_thresh
    mesh_triangles = mesh_triangles[valid_mask]
    mesh_centroids = mesh_centroids[valid_mask]
    # (2) min distance b/w vertex / centroid of a mesh triangle to navmesh
    mesh_tricents = np.concatenate(
        [mesh_triangles, mesh_centroids], axis=1
    )  # (N, 4, 3)
    mesh_tricents = mesh_tricents.reshape(-1, 3)
    _, d2navmesh, _ = navmesh.nearest.on_surface(mesh_tricents)  # (N * 4, )
    d2navmesh = d2navmesh.reshape(-1, 4).min(axis=1)  # (N, )
    closest_mesh_triangles = mesh_triangles[(d2navmesh < closeness_thresh)]
    clutter_area = get_triangle_areas(closest_mesh_triangles).sum().item()
    clutter = clutter_area / (navmesh_area + EPS)

    return clutter

def compute_scene_clutter(
    hsim: habitat_sim.Simulator,
    trimesh_scene: trimesh.parent.Geometry,
    scene_id: str,
    *args: Any,
    closeness_thresh: float = 0.5,
    **kwargs: Any,
) -> float:
    """
    Scene clutter measures amount of clutter in the scene. This is computed as
    the ratio between the raw scene mesh area within 0.5m of the navigable
    regions and the navigable space. We restrict to 0.5m to only pick the
    surfaces that are near navigable spaces in the building
    (e.g., furniture, and interior walls), and to ignore other surfaces outside
    the building. This is implemented in the same way as by Xia et al. to
    make the reported statistics comparable. Higher values are better and
    indicate more cluttered scenes that provide more obstacles for navigation.

    Args:
        hsim: habitat simulator instance
        trimesh_scene: 3D scene loaded in trimesh
        closeness_thresh: a distance threshold for points on the mesh to be
            considered "close" to navigable space.

    Reference:
        Xia, Fei, et al.
        "Gibson env: Real-world perception for embodied agents."
        Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.
    """
    if not hsim.pathfinder.is_loaded:
        return 0.0
    # convert habitat navmesh to a trimesh scene
    total_navmesh_vertices = np.array(hsim.pathfinder.build_navmesh_vertices())
    indoor_islands = kwargs['indoor_islands']
    navigable_area = kwargs['navigable_area']
    outdoor_islands = kwargs['outdoor_islands']
    ## transforming to trimesh not necessary for FP scenes (scene GLBs already have trimesh/standard transform)
    # navmesh_vertices = transform_coordinates_hsim_to_trimesh(navmesh_vertices)
    ## three consecutive vertices form a triangle face
    indoor_navmesh_vertices = [np.array(hsim.pathfinder.build_navmesh_vertices(island_index)) for island_index in indoor_islands]
    indoor_navmesh_vertices = np.concatenate(indoor_navmesh_vertices, axis=0)
    indoor_scene_clutter = compute_scene_clutter_impl(trimesh_scene, indoor_navmesh_vertices, navigable_area['indoor_navigable_area'], closeness_thresh=closeness_thresh, navmesh_id=f'{scene_id}_indoor')
    outdoor_scene_clutter = 0.0
    if len(outdoor_islands):
        outdoor_navmesh_vertices = [np.array(hsim.pathfinder.build_navmesh_vertices(island_index)) for island_index in outdoor_islands]
        outdoor_navmesh_vertices = np.concatenate(outdoor_navmesh_vertices, axis=0)
        outdoor_scene_clutter = compute_scene_clutter_impl(trimesh_scene, outdoor_navmesh_vertices, navigable_area['outdoor_navigable_area'], closeness_thresh=closeness_thresh, navmesh_id=f'{scene_id}_outdoor')
    total_scene_clutter = compute_scene_clutter_impl(trimesh_scene, total_navmesh_vertices, navigable_area['total_navigable_area'], closeness_thresh=closeness_thresh, navmesh_id=f'{scene_id}_total')

    clutter = {
        'indoor_scene_clutter': indoor_scene_clutter,
        'outdoor_scene_clutter': outdoor_scene_clutter,
        'total_scene_clutter': total_scene_clutter,
    }
    return clutter

def compute_floor_area_impl(
    floor_extents: List[Dict[str, float]],
    mesh_vertices: np.ndarray,
    scene_id: str,
    floor_limit: float = 0.5,
    **kwargs: Any,
) -> float:
    # Y (not Z) axis in trimesh is vertically upward for FP scenes
    floor_area = 0.0
    for fext in floor_extents:
        mask = (mesh_vertices[:, 1] >= fext["min"] - floor_limit) & (
            mesh_vertices[:, 1] < fext["max"] + floor_limit
        )
        # floor_convex_hull = ConvexHull(mesh_vertices[mask, :2])
        points = mesh_vertices[mask][:, [0, 2]]
        if len(points) > 0:
            floor_convex_hull = ConvexHull(points)
            visualization = True
            if visualization:
                import matplotlib.pyplot as plt
                fig = plt.figure(figsize=(5, 5))
                ax = fig.add_subplot(111)
                plt.plot(points[:, 0], points[:, 1], '.')
                plt.plot(points[floor_convex_hull.vertices, 0],
                        points[floor_convex_hull.vertices, 1], 'r--', lw=4)
                plt.plot(points[(floor_convex_hull.vertices[-1], floor_convex_hull.vertices[0]), 0],
                        points[(floor_convex_hull.vertices[-1], floor_convex_hull.vertices[0]), 1], 'r--', lw=4)
                plt.plot(points[floor_convex_hull.vertices[:], 0], points[floor_convex_hull.vertices[:], 1],
                        marker='o', markersize=7, color="red")
                ax.set_aspect('equal', adjustable='box')
                os.makedirs('floor-area', exist_ok=True)
                fig.savefig(f'floor-area/{scene_id}.png', dpi=fig.dpi, bbox_inches='tight')
                plt.cla()
            # convex_hull.volume computes the area for 2D convex hull
            floor_area += floor_convex_hull.volume
        else:
            print(f'{scene_id} has 0 floor area')
    return floor_area
    

def compute_floor_area(
    hsim: habitat_sim.Simulator,
    trimesh_scene: trimesh.parent.Geometry,
    scene_id: str,
    floor_limit: float = 0.5,
    **kwargs: Any,
) -> float:
    """
    Floor area (m^2) measures the overall extents of the floor regions in the
    scene. This is the area of the 2D convex hull of all navigable locations in
    a floor. For scenes with multiple floors, the floor space is summed over all
    floors. This is implemented in the same way as by Xia et al. to make the
    reported statistics comparable. Higher values indicate the presence of more
    navigation space and rooms.

    Args:
        hsim: habitat simulator instance
        trimesh_scene: 3D scene loaded in trimesh
        floor_limit: defines the maximum height above the navigable space
            that is considered as a part of the current floor.

    Reference:
        Xia, Fei, et al.
        "Gibson env: Real-world perception for embodied agents."
        Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.
    """
    if not hsim.pathfinder.is_loaded:
        return 0.0
    mesh_vertices = trimesh_scene.triangles.reshape(-1, 3)
    indoor_islands = kwargs['indoor_islands']
    outdoor_islands = kwargs['outdoor_islands']
    indoor_floor_extents = get_floor_navigable_extents(hsim, islands=indoor_islands, num_points_to_sample=2000)
    indoor_floor_area = compute_floor_area_impl(indoor_floor_extents, mesh_vertices, scene_id, floor_limit=floor_limit)
    outdoor_floor_area = 0.0
    if len(outdoor_islands):
        outdoor_floor_extents = get_floor_navigable_extents(hsim, islands=outdoor_islands, num_points_to_sample=2000)
        outdoor_floor_area = compute_floor_area_impl(outdoor_floor_extents, mesh_vertices, scene_id, floor_limit=floor_limit)
    total_floor_extents = get_floor_navigable_extents(hsim, islands=[-1], num_points_to_sample=20000)
    total_floor_area = compute_floor_area_impl(total_floor_extents, mesh_vertices, scene_id, floor_limit=floor_limit)
    floor_area = {
        'indoor_floor_area': indoor_floor_area,
        'outdoor_floor_area': outdoor_floor_area,
        'total_floor_area': total_floor_area,
    }
    return floor_area
