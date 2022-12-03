
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return smoothed

data_key = "test_battle_won_mean"

data = [
    ["results/Formal/PaperExperiments/experiment5/sacred/sc2/", "Fuseformer + Linear Embedding"],
    ["results/Formal/PaperExperiments/experiment6/sacred/sc2/", "Fuseformer + Linear Embedding + Dummy Input"],
    ["results/Formal/PaperExperiments/experiment7/sacred/sc2/", "Fuseformer + XY Embedding + Dummy Input"],
    ["results/Formal/PaperExperiments/experiment8/sacred/sc2/", "Fuseformer + XY Embedding"],
]

mixers = ["VDN", "RMIX"]
maps_name = ["3m", "8m", "5m_vs_6m", "8m_vs_9m", "3s_vs_3z", "3s_vs_5z"]
maps_episode = [1050000, 1050000, 5050000, 2050000, 2050000, 5050000]
maps_smooth_weight = [0.9, 0.9, 0.95, 0.9, 0.9, 0.98]

min_episodes = 0
episode_interval = 10000


##### Embedding types / Dummy embedding #####
for i_map in range(len(maps_name)):
    for i_mixer in range(len(mixers)):

        max_episode = maps_episode[i_map]
        x = np.arange(min_episodes, max_episode + episode_interval, episode_interval)
        y = []

        for d in data:
            path = d[0] + "{}/scratch/{}/info.json".format(maps_name[i_map], i_mixer+1)
            with open(path) as f:
                v = json.load(f)
            y.append(v[data_key])

        plot_x_axis_title = "Episodes"
        plot_y_axis_title = data_key

        plt.figure(figsize = (16, 8))

        for i in range(len(y)):
            tmp = smooth(y[i], maps_smooth_weight[i_map])
            _y = np.asarray(tmp)

            x_len = x.shape[0]
            y_len = _y.shape[0]
            delta = x_len - y_len
            tmp = np.asarray([_y[-1] for _ in range(delta)])
            
            if x_len > y_len:
                _y = np.concatenate([_y, tmp], -1)

            plt.plot(x, _y, label = data[i][1])

        plt.xlabel(plot_x_axis_title)
        plt.ylabel(plot_y_axis_title)
        plt.legend(loc = 'upper left')
        plt.savefig("results/figures/embedding+dummy/{}_{}.png".format(maps_name[i_map], mixers[i_mixer]))
        plt.close()


  
##### Finetune Comparison #####
train_type = ["scratch", "finetune"]
maps_name = ["8m", "8m_vs_9m", "3s_vs_5z"]
maps_episode = [1050000, 2050000, 5050000]
maps_smooth_weight = [0.9, 0.9, 0.98]

for i_map in range(len(maps_name)):

    max_episode = maps_episode[i_map]
    x = np.arange(min_episodes, max_episode + episode_interval, episode_interval)
    y = []

    for i_train_type in range(len(train_type)):
        for d in data:
            path = d[0] + "{}/{}/{}/info.json".format(maps_name[i_map], train_type[i_train_type], i_mixer+1)
            with open(path) as f:
                v = json.load(f)
            y.append(v[data_key])

    plot_x_axis_title = "Episodes"
    plot_y_axis_title = data_key

    plt.figure(figsize = (16, 8))

    for i in range(len(y)):
        tmp = smooth(y[i], maps_smooth_weight[i_map])
        _y = np.asarray(tmp)

        x_len = x.shape[0]
        y_len = _y.shape[0]
        delta = x_len - y_len
        tmp = np.asarray([_y[-1] for _ in range(delta)])
        
        if x_len > y_len:
            _y = np.concatenate([_y, tmp], -1)

        data_len = int(len(data))
        plt.plot(x, _y, label = data[i % data_len][1] + " ({})".format(mixers[i // data_len]))

    plt.xlabel(plot_x_axis_title)
    plt.ylabel(plot_y_axis_title)
    plt.legend(loc = 'upper left')
    plt.savefig("results/figures/mixer/{}.png".format(maps_name[i_map]))
    plt.close()



##### Variants comparison #####
data = [
    ["results/Formal/PaperExperiments/experiment7/sacred/sc2/", "Fuseformer + Linear Embedding + Dummy Input"],
    ["results/Formal/PaperExperiments/experiment9/sacred/sc2/", "Fuseformer++ + Linear Embedding + Dummy Input"],
    ["results/Formal/PaperExperiments/experiment10/sacred/sc2/", "FuseformerExtra + Linear Embedding + Dummy Input"],
    ["results/Formal/PaperExperiments/experiment11/sacred/sc2/", "STformer + Linear Embedding + Dummy Input"],
]
maps_name = ["3m", "8m", "5m_vs_6m", "8m_vs_9m", "3s_vs_3z", "3s_vs_5z"]
maps_episode = [1050000, 1050000, 5050000, 2050000, 2050000, 5050000]
maps_smooth_weight = [0.9, 0.9, 0.95, 0.9, 0.9, 0.98]

for i_map in range(len(maps_name)):
    for i_mixer in range(len(mixers)):

        max_episode = maps_episode[i_map]
        x = np.arange(min_episodes, max_episode + episode_interval, episode_interval)
        y = []

        for d in data:
            path = d[0] + "{}/scratch/{}/info.json".format(maps_name[i_map], i_mixer+1)

            if os.path.exists(path):
                with open(path) as f:
                    v = json.load(f)
                y.append(v[data_key])
            else:
                print("{} not exist".format(path))
                y.append([0])

        plot_x_axis_title = "Episodes"
        plot_y_axis_title = data_key

        plt.figure(figsize = (16, 8))

        for i in range(len(y)):
            tmp = smooth(y[i], maps_smooth_weight[i_map])
            _y = np.asarray(tmp)

            x_len = x.shape[0]
            y_len = _y.shape[0]
            delta = x_len - y_len
            tmp = np.asarray([_y[-1] for _ in range(delta)])
            
            if x_len > y_len:
                _y = np.concatenate([_y, tmp], -1)

            plt.plot(x, _y, label = data[i][1])

        plt.xlabel(plot_x_axis_title)
        plt.ylabel(plot_y_axis_title)
        plt.legend(loc = 'upper left')
        plt.savefig("results/figures/variants/{}_{}.png".format(maps_name[i_map], mixers[i_mixer]))
        plt.close()