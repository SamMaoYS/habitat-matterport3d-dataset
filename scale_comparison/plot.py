import pandas as pd
import pdb
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    hm3d_df = pd.read_csv('data/hm3d_metrics_stage.csv', sep='\t')
    print(hm3d_df[hm3d_df['indoor_navigable_area'] == 0])
    hm3d_df = hm3d_df[hm3d_df['indoor_navigable_area'] > 10]
    fp_df = pd.read_csv('data/FloorPlanner_metrics.csv')
    print(fp_df[fp_df['indoor_navigable_area'] == 0])
    fp_df = fp_df[fp_df['indoor_navigable_area'] > 50]
    procthor_df = pd.read_csv('data/ProcTHOR_metrics.csv')
    print(procthor_df[procthor_df['indoor_navigable_area'] == 0])
    procthor_df = procthor_df[procthor_df['indoor_navigable_area'] > 10]

    procthor_df['dataset'] = 'ProcTHOR'
    fp_df['dataset'] = 'HSSD-200'
    hm3d_df['dataset'] = 'HM3DSem'
    df = pd.concat([procthor_df, fp_df, hm3d_df], ignore_index=True)
    df.to_csv('navarea_dist.csv')
    # sns.histplot(df, x="indoor_navigable_area", hue="dataset", element="poly", stat='density', common_norm=False, kde=True)
    # sns.set_theme(style='white')
    hue_order = ['ProcTHOR', 'HSSD-200', 'HM3DSem']
    sns.set(style="white", font_scale=1.5)
    # ax = sns.kdeplot(df, x="indoor_navigable_area", hue="dataset", bw_adjust=0.5, hue_order=hue_order, fill=True, common_norm=False, alpha=0.05, linewidth=2, cut=3, clip=(0, 600.0), palette=["C0", "C1", "C3", "k"])
    ax = sns.kdeplot(df, x="indoor_navigable_area", hue="dataset", bw_adjust=0.5, hue_order=hue_order, fill=False, common_norm=False, alpha=1.00, linewidth=2, cut=3, clip=(0, 600.0), palette=["C0", "C1", "C3", "k"])
    # ax.set_xlim(0, 400)
    sns.despine()
    plt.legend(title=None, loc='upper right', labels=['HM3DSem', 'HSSD-200', 'ProcTHOR'], reverse=True)
    # ax.add_legend(label_order = sorted(ax._legend_data.keys(), key = int))
    ax.set(xlabel='Navigable Area ($m^2$)')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(2,-2))
    plt.tight_layout()
    plt.savefig(f'navarea_dist_fill.pdf', bbox_inches='tight')
    plt.savefig(f'navarea_dist_fill.png', bbox_inches='tight')
    plt.show()
