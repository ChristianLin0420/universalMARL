from asyncio import sleep
from tkinter.messagebox import NO
import numpy as np
import os
import time
import collections
import threading
from pynvml import *
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml

from run import run
from envs.smac_config import get_smac_map_config

SETTINGS['CAPTURE_MODE'] = "fd" # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("universalMARL")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")

PAUSE_NEXT_TASK_DURATION = 0

BASELINES_MODEL_PATH = []
BENCHMARKS_MODEL_PATH = []

@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]

    # run the framework
    info = run(_run, config, _log)

    # record paths from baselines/benchmarks model training
    path = info["saved_model_path"]

    if config["agent"] != "rnn":
        if config["task_dir"] == "baselines":
            BASELINES_MODEL_PATH.append(path)
        elif config["task_dir"] == "benchmarks":
            BENCHMARKS_MODEL_PATH.append(path)


def _get_config_name(params, arg_name, delete = True):
    config_name = None

    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]

            if delete:
                del params[_i]

            break;

    return config_name

def _get_config(params, arg_name, subfolder, extra = None):

    if extra is None:
        config_name = _get_config_name(params, arg_name)
    else:
        config_name = extra

    assert None != config_name

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                # config_dict = yaml.load(f)
                config_dict = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict
        


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)

def auto(params):

    default_config_name = "default.yaml"

    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", default_config_name), "r") as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # automatically play entire experiments (set experiment config)
    ex_config = _get_config(params, "", "experiments", config_dict["env"])

    parallel_env = "_beta" if ex_config["parallel"] and ex_config["env"] == "sc2" else ""
    mixing_networks = ex_config["mixing_networks"]
    agent_models = ex_config["agent_models"]

    cuda_available = th.cuda.is_available()
    initialization = True

    if cuda_available:
        config_dict["use_cuda"] = True
        nvmlInit()

    # load environment config
    env_config = _get_config(params, "", "envs", config_dict["env"] + parallel_env)

    if config_dict["env"] == "sc2":
        map_c = get_smac_map_config(config_dict["map_name"])
        env_config = recursive_dict_update(env_config, map_c)
    # elif config_dict["env"] == "simple_spread":
    #     u = { "env_args": { "n_agents": val_s[0], "n_landmarks" : val_s[1] } }
    #     env_config = recursive_dict_update(env_config, u)

    for key_a in agent_models:
        agent = { "agent": key_a }
        config_dict = recursive_dict_update(config_dict, agent)

        for m_net in mixing_networks:
            # get algorithm config
            alg_config = _get_config(params, "--config", "algs", m_net + parallel_env)

            config_dict = recursive_dict_update(config_dict, env_config)
            config_dict = recursive_dict_update(config_dict, alg_config)

            config_dict["task_dir"] = "scratch" if config_dict["checkpoint_path"] == "" else "finetune"

            logger = get_logger()
            ex.logger = logger

            # Save to disk by default for sacred
            logger.info("Saving to FileStorageObserver in results/sacred.")
            file_obs_path = os.path.join(results_path, config_dict["experiment"], "sacred", config_dict["env"], config_dict["map_name"], config_dict["task_dir"])

            # now add all the config to sacred
            if initialization:
                ex.add_config(config_dict)
                ex.observers.append(FileStorageObserver.create(file_obs_path))
                ex.run()

                initialization = False
            else:
                ex.observers[0] = FileStorageObserver.create(file_obs_path)
                ex.run(config_updates = config_dict)
            
            time.sleep(PAUSE_NEXT_TASK_DURATION)

def single(params):
    # environment checking
    USING_MPS = False

    # environment checking
    # if USING_MPS:
    #     import torch
    #     print(f"PyTorch version: {torch.__version__}")

    #     # Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
    #     print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
    #     print(f"Is MPS available? {torch.backends.mps.is_available()}")

    #     # Set the device      
    #     device = "mps" if torch.backends.mps.is_available() else "cpu"
    #     print(f"Using device: {device}")

    default_config_name = _get_config_name(params, "--env-config")
    default_config_name = "default_{}.yaml".format(default_config_name)

    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", default_config_name), "r") as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    if default_config_name != "default_sc2.yaml":
        env_config = _get_config(params, "--scenario", "envs")
    else:
        env_config = _get_config(params, "--env-config", "envs", "sc2")

    alg_config = _get_config(params, "--config", "algs")

    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    config_dict["task_dir"] = "default"
    config_dict["experiment"] = "test"

    # now add all the config to sacred
    ex.add_config(config_dict)

    logger = get_logger()
    ex.logger = logger

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path,  config_dict["experiment"], "sacred", config_dict["env"], config_dict["map_name"], config_dict["task_dir"])
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline(params)

if __name__ == '__main__':

    params = deepcopy(sys.argv)

    if len(params) == 1: 
        auto(params)
    else:
        single(params)