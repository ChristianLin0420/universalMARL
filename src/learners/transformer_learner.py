import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.transMix import TransMixer
from modules.mixers.adaptivemix import AdaptiveMixer
import torch as th
import torch.nn as nn
from torch.optim import RMSprop
from random import shuffle


class TransLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        if not args.random_mixing_inputs_zero:
            self.mixing_query = th.unsqueeze(th.rand(args.max_mixing_size, args.token_dim + 1), 0)
        else:
            self.mixing_query = th.unsqueeze(th.zeros(args.max_mixing_size, args.token_dim + 1), 0)

        self.mixing_query = nn.Parameter(self.mixing_query)

        self.mixing_indices = [i for i in range(args.max_mixing_size)]

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "transmix":
                self.mixer = TransMixer(args)
            if args.mixer == "adaptivemix":
                self.mixer = AdaptiveMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def expand_inputs(self, inputs, randomize = False):
        b, t, a, e = inputs.size()
        new_inputs = th.repeat_interleave(self.mixing_query, t, dim = 0).unsqueeze(0)
        new_inputs = th.repeat_interleave(new_inputs, b, dim = 0)            

        if not randomize:
            new_inputs[:, :, :a, :] = inputs
        else:
            shuffle(self.mixing_indices)
            
            for i, idx in enumerate(self.mixing_indices[:self.args.ally_num]):
                new_inputs[:, :, idx:idx+1, :] = inputs[:, :, i:i+1, :]

        if self.args.use_cuda:
            new_inputs.cuda()

        return new_inputs

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        observations = batch["obs"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions)  # Remove the last dim
        
        if self.args.mixer == "transmix":
            chosen_action_qvals = th.cat([chosen_action_qvals, observations[:, :, :, :5]], dim = 3)
            chosen_action_qvals = self.expand_inputs(chosen_action_qvals, self.args.random_inputs)

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions)

            if self.args.mixer == "transmix":
                target_max_qvals = th.cat([target_max_qvals, observations[:, :, :, :5]], dim = 3)
                target_max_qvals = self.expand_inputs(target_max_qvals, self.args.random_inputs)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1], observations[:, :, :, :5])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], observations[:, :, :, :5])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixing_query.cuda()
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
            th.save(self.mixing_query, "{}/mixing_query.pt".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))

            if self.args.mixer == "adaptivemix":
                self.mixer.new_adaptive_layer()

        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
        self.mixing_query = th.load("{}/mixing_query.pt".format(path), map_location=lambda storage, loc: storage)
        self.mixing_query.requires_grad =  False

