
import numpy as np

from components.madt_buffer import ReplayBuffer
from components.rollout_worker import RolloutWorker

from envs import REGISTRY as env_REGISTRY

from envs.smac_config import get_map_params
from components.madt_config import get_config
from runners.wrappers import ShareSubprocVecEnv


class Env:
    def __init__(self, n_threads = 1, args = None):
        parser = get_config()
        all_args = parser.parse_known_args()[0]
        self.real_env = ShareSubprocVecEnv([env_REGISTRY[args.env] for _ in range(n_threads)])
        self.num_agents = get_map_params(all_args.map_name)["n_agents"]
        self.max_timestep = get_map_params(all_args.map_name)["limit"]
        self.n_threads = n_threads


class MADTRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        self.eval_env = Env(args.batch_size_run, args)
        self.online_train_env = Env(args.batch_size)     

        self.log_train_stats_t = -100000

        self.target_rtgs = 20.

    def setup(self, mac):

        block_size = self.args.context_length * 3
        global_obs_dim = self.args.vocab_size
        local_obs_dim = self.args.state_size
        action_dim = self.args.action_size

        self.mac = mac
        self.buffer = ReplayBuffer(block_size, global_obs_dim, local_obs_dim, action_dim)
        self.rollout_worker = RolloutWorker(self.mac.actor, self.mac.critic, self.buffer, global_obs_dim, local_obs_dim, action_dim)

        # prepare the buffer
        used_data_dir = []
        for map_name in ['3s_vs_4z', '2m_vs_1z', '3m', '2s_vs_1sc', '3s_vs_3z']:
            source_dir ="src/offline_data/" + map_name + "_good_1000/1/"
            used_data_dir.append(source_dir)
        
        self.buffer.load_offline_data(used_data_dir, [200, 200, 200, 200, 200], max_epi_length = self.eval_env.max_timestep)
        offline_dataset = self.buffer.sample()
        offline_dataset.stats()

        self.rollout_worker = RolloutWorker(mac.actor, mac.critic, self.buffer, global_obs_dim, local_obs_dim, action_dim)

    def close_env(self):
        self.eval_env.real_env.close()
        self.online_train_env.close()

    def reset(self):
        self.t = 0
        self.env_steps_this_run = 0

    def sample_dataset(self):
        dataset = self.buffer.sample()
        dataset.stats()

        return dataset

    def run(self, test_mode = False):

        cur_stats = {}

        if test_mode:
            returns, battle_won = self.rollout_worker.rollout(self.eval_env, self.target_rtgs, train=False)
            cur_stats["battle_won"] = np.mean(battle_won)

            self._log(returns, cur_stats, "test_", battle_won)
        else:
            returns, battle_won = self.rollout_worker.rollout(self.online_train_env, self.target_rtgs, train=True)
            cur_stats["battle_won"] = np.mean(battle_won)
            
            if self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
                self._log(returns, cur_stats, "")
                self.log_train_stats_t = self.t_env

    def _log(self, returns, stats, prefix, test_win = None):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()

        if test_win is not None:
            env_avg_win_rate = np.reshape(test_win, (self.batch_size, self.args.test_nepisode // self.batch_size))
            env_avg_win_rate = np.mean(env_avg_win_rate, axis = 1)
            self.logger.log_stat(prefix + "win_rate_std", np.std(env_avg_win_rate), self.t_env)
            test_win.clear()