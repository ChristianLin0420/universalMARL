
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

experiment_index = "0"
sacred_indices = [2, 3, 12]
data_paths = ["results/experiment {}/sacred/navigation/{}/info.json".format(experiment_index, idx) for idx in sacred_indices]
data_labels = ["VDN", "QMix", "Qtran"]
data_titles = [ "RNN-based agent with various mixing network(5 agents) Reward",
                "RNN-based agent with various mixing network(5 agents) Loss"]
data_keys = ["return_mean", "loss"]
data_y_label = ["Reward", "Loss"]

min_episodes = 0
max_episode = 300000
episode_interval = 10000

x = np.arange(min_episodes, max_episode + episode_interval, episode_interval)
datas = []

for path in data_paths:
    with open(path) as f:
        data = json.load(f)
    datas.append(data)

for idx in range(len(data_keys)):
    plot_name = data_titles[idx] + ".png"
    plot_title = data_titles[idx]
    plot_x_axis_title = "Episodes"
    plot_y_axis_title = data_y_label[idx]

    plt.figure(figsize = (7, 24))
    
    for i, d in enumerate(datas):
        if isinstance(d[data_keys[idx]][0], dict):
            y = []

            for val in d[data_keys[idx]]:
                y.append(val["value"])
            
            y = np.asarray(y)
        elif isinstance(d[data_keys[idx]][0], list):
            y = np.asarray(d[data_keys[idx]])
        
        plt.plot(x, y, label = data_labels[i])

    plt.title(plot_title)
    plt.xlabel(plot_x_axis_title)
    plt.ylabel(plot_y_axis_title)
    plt.legend(loc = 'best')
    plt.savefig(plot_name)
    plt.close()
