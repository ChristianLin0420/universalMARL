from aifc import Error
from modules.agents import REGISTRY as agent_REGISTRY, TRANSFORMERbasedAgent
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
import torch.nn as nn


# This multi-agent controller shares parameters between agents
class BasicMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

        if args.agent in ["fuseformer++", "fuseformer_extra", "stformer", "stformer++"]:
            m_factor = 1
            e_factor = 1

            if args.agent in ["stformer"]:
                m_factor = 2
            elif args.agent in ["fuseformer_extra", "stformer++"]:
                m_factor = 2
                e_factor = 2

            self.decoder_outputs_test = th.zeros(args.ally_num, args.max_memory_decoder * m_factor, args.emb * e_factor).to(args.device)
            self.decoder_outputs_train = th.zeros(args.ally_num * args.batch_size, args.max_memory_decoder * m_factor, args.emb * e_factor).to(args.device)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):

        # rnn based agent
        if self.args.agent not in TRANSFORMERbasedAgent:
            agent_inputs = self._build_inputs(ep_batch, t)
            avail_actions = ep_batch["avail_actions"][:, t]
            agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

            # Softmax the agent outputs if they're policy logits
            if self.agent_output_type == "pi_logits":

                if getattr(self.args, "mask_before_softmax", True):
                    # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                    reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                    agent_outs[reshaped_avail_actions == 0] = -1e10

                agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
                if not test_mode:
                    # Epsilon floor
                    epsilon_action_num = agent_outs.size(-1)
                    if getattr(self.args, "mask_before_softmax", True):
                        # With probability epsilon, we will pick an available action uniformly
                        epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                    agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                                   + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                    if getattr(self.args, "mask_before_softmax", True):
                        # Zero out the unavailable actions
                        agent_outs[reshaped_avail_actions == 0] = 0.0

        # transformer based agent
        else:
            agent_inputs = self._build_inputs_transformer(ep_batch, t, self.args.env)

            if self.args.env == "sc2":

                hidden_size = self.args.emb // 2 if self.args.agent == "trackformer" else self.args.emb

                if self.args.agent == "double_perceiver":
                    hidden_state = None
                else:
                    hidden_state = self.hidden_states.reshape(-1, 1, hidden_size)
                
                if self.args.agent in ["fuseformer++", "fuseformer_extra", "stformer", "stformer++"]:
                    
                    shift = 1

                    if self.args.agent == "stformer":
                        shift = 2

                    if agent_inputs.size(0) == self.args.ally_num:
                        self.decoder_outputs_train = th.mul(self.decoder_outputs_train, 0.0)
                        agent_outs, self.hidden_states, decoder_outs = self.agent(agent_inputs,
                                                                                  hidden_state,
                                                                                  self.decoder_outputs_test,
                                                                                  self.args.enemy_num, self.args.ally_num)
                        decoder_outs = decoder_outs.detach()
                        self.decoder_outputs_test = th.cat((decoder_outs, self.decoder_outputs_test[:, :-shift, :]), 1)
                    elif agent_inputs.size(0) == self.args.ally_num * self.args.batch_size:
                        self.decoder_outputs_test = th.mul(self.decoder_outputs_test, 0.0)
                        agent_outs, self.hidden_states, decoder_outs = self.agent(agent_inputs,
                                                                                  hidden_state,
                                                                                  self.decoder_outputs_train,
                                                                                  self.args.enemy_num, self.args.ally_num)
                        decoder_outs = decoder_outs.detach()
                        self.decoder_outputs_train = th.cat((decoder_outs, self.decoder_outputs_train[:, :-shift, :]), 1)
                    else:
                        Error("wrong inputs")
                else:
                    agent_outs, self.hidden_states = self.agent(agent_inputs,
                                                                hidden_state,
                                                                self.args.enemy_num, self.args.ally_num)

            else:
                agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states.reshape(-1, 1, self.args.emb), env = self.args.env)
        
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        if self.args.agent not in TRANSFORMERbasedAgent:
            self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
        else:
            self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, 1, -1)


    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

        if self.args.agent in ["transfermer", "transfermer++", "perceiver_io", "perceiver++", "double_perceiver", "fuseformer"]:
            self.agent.save_query(path)

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

        if self.args.agent in ["transfermer", "transfermer++", "gpt", "perceiver_io", "perceiver++", "double_perceiver", "updet", "fuseformer", "fuseformer++", "fuseformer_extra", "stformer", "stformer++"]:
            # self.agent.load_query(path)
            self.agent.fixed_models_weight()

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _build_inputs_transformer(self, batch, t, env = "sc2"):
        # currently we only support battles with marines (e.g. 3m 8m 5m_vs_6m)
        # you can implement your own with any other agent type.
        inputs = []
        raw_obs = batch["obs"][:, t]
        rearranged_idx = self.args.own_feature
        arranged_obs = th.cat((raw_obs[:, :, -rearranged_idx:], raw_obs[:, :, :-rearranged_idx]), 2)

        if env == "sc2":
            token_size = self.args.token_dim
            reshaped_obs = arranged_obs.view(-1, 1 + (self.args.enemy_num - 1) + self.args.ally_num, token_size)
        elif env == "simple_spread":
            reshaped_obs = arranged_obs.view(-1, 1 + (self.args.env_args["n_agents"] - 1), self.args.token_dim)

        inputs.append(reshaped_obs)
        inputs = th.cat(inputs, dim=1)

        return inputs.to(self.args.device)

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape