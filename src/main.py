from tkinter.messagebox import NO
import numpy as np
import os
import collections
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

SETTINGS['CAPTURE_MODE'] = "fd" # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("universalMARL")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]

    # run the framework
    run(_run, config, _log)

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
        config_name = "sc2"

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

USING_MPS = False

if __name__ == '__main__':

    # environment checking
    if USING_MPS:
        import torch
        print(f"PyTorch version: {torch.__version__}")

        # Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
        print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
        print(f"Is MPS available? {torch.backends.mps.is_available()}")

        # Set the device      
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {device}")

    params = deepcopy(sys.argv)

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

    # now add all the config to sacred
    ex.add_config(config_dict)

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path, "sacred", config_dict["env"])
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline(params)

