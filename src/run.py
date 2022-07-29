import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot


def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda:{}".format(args.gpu_id) if args.use_cuda else "cpu"

    if args.use_cuda:
        th.cuda.set_device(args.device)

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent = 4, width = 1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", args.experiment, "tb_logs", args.env, args.task_dir)
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        print("tb_exp_direc: {}".format(tb_exp_direc))
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    info = {}

    if args.agent == "madt":
        madt_run(args = args, logger = logger)
    elif args.agent == "meta":
        meta_run(args = args, logger = logger)
    else:
        info = run_sequential(args = args, logger = logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    # os._exit(os.EX_OK)

    return info


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args = args, logger = logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }

    groups = {
        "agents": args.n_agents
    }

    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device = "cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme = scheme, groups = groups, preprocess = preprocess, mac = mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key = lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)

        if args.finetuned:
            runner.t_env = 0
        else:
            runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    info = {}

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time
        episode_batch = runner.run(test_mode = False)
        buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner.train(episode_sample, runner.t_env, episode)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            # model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, args.experiment, "models", args.env, args.task_dir, args.unique_token, str(runner.t_env))
            #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok = True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

            if runner.t_env - model_save_time >= args.save_model_interval:
                info["saved_model_path"] = save_path

            model_save_time = runner.t_env

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")

    return info

def meta_run(args, logger):
    pass

def madt_run(args, logger):
    # Init runner so we can get env info
    runner = r_REGISTRY["madt"](args = args, logger = logger)  

    # Setup multiagent controller here
    mac = mac_REGISTRY["madt_mac"](args)

    # Give runner the scheme
    runner.setup(mac = mac)

    # Learner
    class MADTConfig:
        # optimization parameters
        max_epochs = 1
        batch_size = 32
        learning_rate = 5e-4
        betas = (0.9, 0.95)
        grad_norm_clip = 0.5
        weight_decay = 0.1  # only applied on matmul weights
        # checkpoint settings
        num_workers = 0  # for DataLoader

        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    offline_config = MADTConfig(num_workers = args.batch_size_run, mode = "offline")
    offline_learner = le_REGISTRY["madt_learner"](mac, args, offline_config, logger)
    offline_dataset = runner.sample_dataset(False)
    
    if args.use_cuda:
        offline_learner.cuda()

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))
    save_path = ""
    
    # offline training
    for i in range(10):
        _ = offline_learner.train(offline_dataset, True, True)

        # save the models
        save_path = os.path.join(args.local_results_path, args.experiment, "models", args.env, args.task_dir, args.unique_token, "offline")

        os.makedirs(save_path, exist_ok = True)
        logger.console_logger.info("Saving models to {}".format(save_path))

        offline_learner.save_models(save_path)


    # online training 
    online_config = MADTConfig(num_workers = args.batch_size_run, mode = "online")
    online_learner = le_REGISTRY["madt_learner"](mac, args, online_config, logger)
    online_learner.load_models(save_path)
    
    if args.use_cuda:
        online_learner.cuda()

    while runner.t_env <= args.t_max:
        
        runner.run(test_mode = False)

        online_dataset = runner.sample_dataset()
        runner.t_env = online_learner.train(online_dataset, True, False)

        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))

            last_time = time.time()
            last_test_T = runner.t_env

            runner.run(test_mode = True)

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            save_path = os.path.join(args.local_results_path, args.experiment, "models", args.env, args.task_dir, args.unique_token, str(runner.t_env))

            os.makedirs(save_path, exist_ok = True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            online_learner.save_models(save_path)
            model_save_time = runner.t_env

        episode += args.batch_size_run

    runner.close_env()
    logger.console_logger.info("Finished Training")

def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"] // config["batch_size_run"]) * config["batch_size_run"]

    return config
