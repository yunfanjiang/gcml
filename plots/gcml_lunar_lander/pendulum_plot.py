from typing import List

import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
# sns.set(font_scale=1.15)
import matplotlib.pyplot as plt


def smooth(
    scalars: List[float], weight: float
) -> List[float]:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return smoothed


def normalize_score(x, max):
    print(type(max))
    print(type(x))
    return (max - x) / max


if __name__ == "__main__":
    N = 201
    files = [
        'run-pendulum_outer_lr_0.001_inner_lr_0.4_learn_inner_lr_True_use_task_config_True_n_tasks_20_traj_len_50_run0-tag-train_distance_post_adapt_query.csv',
        'run-pendulum_outer_lr_0.001_inner_lr_0.4_learn_inner_lr_True_use_task_config_True_n_tasks_20_traj_len_50_run1-tag-train_distance_post_adapt_query.csv',
        'run-pendulum_outer_lr_0.001_inner_lr_0.4_learn_inner_lr_True_use_task_config_True_n_tasks_20_traj_len_50_run2-tag-train_distance_post_adapt_query.csv',
    ]
    ys, xs = [], []
    for file in files:
        df = pd.read_csv(file)
        df = df[:N]
        ys.append(normalize_score(smooth(df['Value'].to_numpy(), 0.97), max=np.float64(np.pi)))
        xs.append(df['Step'].to_numpy())
    x = np.stack(xs, axis=0)
    y = np.stack(ys, axis=0)
    x = np.mean(x, axis=0)
    y_mean = np.mean(y, axis=0)
    y_std = np.std(y, axis=0)

    f, axarr = plt.subplots(2, sharex=True)

    sns.lineplot(x=x, y=y_mean, label='GCML', ax=axarr[0])
    upper = y_mean + y_std
    lower = y_mean - y_std
    axarr[0].fill_between(x, lower, upper, alpha=.3)

    ppo = [normalize_score(1.56, max=np.float64(np.pi))] * len(x)
    sns.lineplot(x=x, y=ppo, label='PPO', ax=axarr[1], color='r', linestyle='--')

    plt.legend(loc='best')
    axarr[1].set_xlabel('Number of Iterations')
    axarr[0].set_ylabel('Normalized Distance Score')
    axarr[1].set_ylabel('Normalized Distance Score')
    plt.tight_layout()
    f.savefig('pendulum_plot.pdf')
    plt.show()
