import os
import argparse
import numpy as np
import pandas as pd
from enum import Enum
from glob import glob
import pdb

class Dataset(Enum):
    iTHOR = 'iTHOR'
    RoboTHOR = 'RoboTHOR'
    ArchitecTHOR = 'ArchitecTHOR'
    ProcTHOR = 'ProcTHOR'
    FloorPlanner = 'FloorPlanner'

# def main(args):
#     tmp_df = pd.read_csv(args.input, sep="\t")
#     df_list = []
#     if args.dataset == Dataset.iTHOR.value:
#         for i, row in tmp_df.iterrows():
#             scene_name = row['scene']
#             scene_idx = int(scene_name.replace('_physics', '').replace('FloorPlan', ''))
            
#             if scene_idx < 21:
#                 split = 'train'
#             elif 21 <= scene_idx < 26:
#                 split = 'val'
#             elif 26 <= scene_idx < 31:
#                 split = 'test'
#             if scene_idx > 500:
#                 split = ''
#             row['split'] = split
#             df_list.append(row.to_frame().T)
#     elif args.dataset in [Dataset.RoboTHOR.value, Dataset.ArchitecTHOR.value]:
#         for i, row in tmp_df.iterrows():
#             scene_name = row['scene']
#             if 'Train' in scene_name:
#                 split = 'train'
#             elif 'Val' in scene_name:
#                 split = 'val'
#             elif 'Test' in scene_name:
#                 split = 'test'
#             row['split'] = split
#             df_list.append(row.to_frame().T)
    
#     df = pd.concat(df_list, ignore_index=True)
#     df = df[df['split'].isin(['train', 'val', 'test'])]
#     df.to_csv(os.path.join(args.output_dir, args.dataset + '_stats.csv'))
#     for column in ['navigable_area', 'navigation_complexity', 'scene_clutter', 'floor_area']:
#         df[column] = pd.to_numeric(df[column])
#     mean_df = df.groupby('split').mean(numeric_only=True)
#     sum_df = df.groupby('split').sum(numeric_only=True)
#     mean_str = mean_df.to_string()
#     sum_str = sum_df.to_string()
#     num_scenes = len(df)
#     output_str = '\n'.join([
#         f'Number of scenes: {num_scenes}',
#         'Mean by split',
#         mean_str,
#         'Sum by split',
#         sum_str,
#         'Mean',
#         df.mean(numeric_only=True).to_string(),
#         'Sum',
#         df.sum(numeric_only=True).to_string(),
#     ])
#     with open(os.path.join(args.output_dir, args.dataset + '_stats.txt'), "w+") as fp:
#         fp.write(output_str)

def main(args):
    metric_files = glob(os.path.join(args.input_dir, '*.csv'))
    df_list = []
    for metric_file in metric_files:
        tmp_df = pd.read_csv(metric_file, sep="\t")
        # scene_name = tmp_df['scene'][0]
        # scene_idx = int(scene_name.replace('_physics', '').replace('FloorPlan', ''))
        # if scene_idx >= 500:
        #     continue
        df_list.append(tmp_df)
    df = pd.concat(df_list, ignore_index=True)
    df.to_csv(os.path.join(args.output_dir, args.dataset + '_metrics.csv'), index=False)
    all_scenes = np.loadtxt(args.scenes, dtype=str).astype(str)
    processed_scenes = df.scene.unique().tolist()
    unprocessed_scenes = np.setdiff1d(all_scenes, processed_scenes)
    print(f'{len(unprocessed_scenes)} unprocessed scenes')
    print(unprocessed_scenes)
    # pdb.set_trace()
    # columns = df.columns.tolist()
    # columns = columns.remove('scene')
    # for column in ['navigable_area', 'navigation_complexity', 'scene_clutter', 'floor_area']:
    #     df[column] = pd.to_numeric(df[column])
    num_scenes = len(df)
    output_str = '\n'.join([
        f'Number of scenes: {num_scenes}',
        'Mean',
        df.mean(numeric_only=True).to_string(),
        'Median',
        df.median(numeric_only=True).to_string(),
        'Sum',
        df.sum(numeric_only=True).to_string(),
    ])
    print(output_str)
    with open(os.path.join(args.output_dir, args.dataset + '_stats.txt'), "w+") as fp:
        fp.write(output_str)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--scenes', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()

    main(args)