import torch as th
import torch.nn as nn
import torch.nn.functional as F

from aifc import Error
from modules.helpers.embedding.positional_embedding import PositionalEncoding

class GMixer(nn.Module):
    def __init__(self, args):
        super(GMixer, self).__init__()

        self.args = args
        self.n_agents = args.max_ally_num
        self.input_dim = args.max_states_dim // 4
        self.emb_dim = 8 #self.args.emb // 2

        self.position_embedding = PositionalEncoding(self.emb_dim, args.max_memory_decoder, args.device)
        self.pos = self.position_embedding.generate()

        self.qs_map_network = nn.Linear(args.max_ally_num, self.emb_dim, bias=False)
        self.state_map_network = nn.Linear(args.max_states_dim, self.input_dim)
        self.tokeys = nn.Linear(self.input_dim, self.emb_dim ** 2, bias=False)
        self.toqueries = nn.Linear(self.input_dim, self.emb_dim ** 2, bias=False)
        self.tovalues = nn.Linear(self.input_dim, self.emb_dim ** 2, bias=False)

        self.hyper_w_final = nn.Sequential(
            nn.Linear(self.input_dim, self.args.hypernet_embed),
            nn.ReLU(),
            nn.Linear(self.args.hypernet_embed, self.emb_dim)
        )

        self.V = nn.Sequential(
            nn.Linear(self.input_dim, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, 1)
        )

        self.q_value_map_network = nn.Linear(self.emb_dim, 1)

        self.outputs_test = th.zeros(1, args.max_memory_decoder, self.emb_dim).to(args.device)
        self.outputs_train = th.zeros(args.batch_size, args.max_memory_decoder, self.emb_dim).to(args.device)


    def state_padding(self, states):

        padding_size = self.args.max_states_dim - states.size(2)
        padding = th.zeros(states.size(0), states.size(1), padding_size).to(self.args.device)
        states = th.cat([states, padding], axis = -1)
        
        return states.view(-1, self.args.max_states_dim)

    def agent_qs_padding(self, agent_qs):
        
        # padding_size = self.args.max_ally_num - agent_qs.size(2)
        padding_size = self.emb_dim - agent_qs.size(2)
        padding = th.zeros(agent_qs.size(0), agent_qs.size(1), padding_size).to(self.args.device)
        agent_qs = th.cat([agent_qs, padding], axis = -1)

        # return agent_qs.view(-1, 1, self.args.max_ally_num)
        return agent_qs.view(-1, 1, self.emb_dim)

    def forward(self, agent_qss, statess):
        b, t, e = agent_qss.size()
        q_tots = None

        for i in range(t):
            agent_qs = agent_qss[:, i:i+1, :]
            states = statess[:, i:i+1, :]

            states = self.state_padding(states)
            agent_qs = self.agent_qs_padding(agent_qs)
            # agent_qs = self.qs_map_network(agent_qs)
            agent_qs = agent_qs.detach()

            # map states to lower dimension
            states = self.state_map_network(states)

            keys = self.tokeys(states).view(-1, self.emb_dim, self.emb_dim)
            queries = self.toqueries(states).view(-1, self.emb_dim, self.emb_dim)
            values = self.tovalues(states).view(-1, self.emb_dim, self.emb_dim)

            if b == 1:
                self.outputs_train = th.mul(self.outputs_train, 0.0)
                self.outputs_test = th.cat((agent_qs, self.outputs_test[:, :-1, :]), 1)
                self.outputs_test = self.outputs_test + self.pos

                keys = th.bmm(self.outputs_test, keys)
                queries = th.bmm(self.outputs_test, queries)
                values = th.bmm(self.outputs_test, values)
            elif b == self.args.batch_size:
                self.outputs_test = th.mul(self.outputs_test, 0.0)
                self.outputs_train = th.cat((agent_qs, self.outputs_train[:, :-1, :]), 1)
                self.outputs_train = self.outputs_train + self.pos

                keys = th.bmm(self.outputs_train, keys)
                queries = th.bmm(self.outputs_train, queries)
                values = th.bmm(self.outputs_train, values)
            else:
                Error("wrong inputs")

            queries = queries / (e ** (1 / 4))
            keys = keys / (e ** (1 / 4))
            dot = th.bmm(queries, keys.transpose(1, 2))

            assert dot.size() == (b, self.args.max_memory_decoder, self.args.max_memory_decoder)

            dot = F.softmax(dot, dim=2)
            out = th.bmm(dot, values).view(b, self.args.max_memory_decoder, self.emb_dim)
            out = F.softmax(out, dim=2)
            hidden = out.transpose(1, 2).contiguous().view(b, self.args.max_memory_decoder, self.emb_dim)

            # w_final = th.abs(self.hyper_w_final(states))
            # w_final = w_final.view(-1, self.emb_dim, 1)
            
            # v = self.V(states).view(-1, 1, 1)
            # y = th.bmm(hidden[:, :1, :], w_final) + v
            # q_tot = y.view(b, 1, 1)

            q_tot = self.q_value_map_network(hidden[:, :1, :]).view(b, 1, 1)

            if q_tots is None:
                q_tots = q_tot
            else:
                q_tots = th.cat([q_tot, q_tots], axis = 1)

        return q_tots

    def fixed_models_weight(self):
        self.position_embedding.requires_grad = False
        self.qs_map_network.requires_grad = False
        self.state_map_network.requires_grad = True
        self.tokeys.requires_grad = False
        self.toqueries.requires_grad = False
        self.tovalues.requires_grad = False
        self.hyper_w_final.requires_grad = False
        self.V.requires_grad = False
        self.q_value_map_network.requires_grad = False

