import torch.nn as nn
import torch

from modules.helpers.models.perceiverIO import PerceiverIO


class PerceiverIOplusAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(PerceiverIOplusAgent, self).__init__()

        self.args = args
        self.action_query = nn.Parameter(torch.rand(1, args.token_dim)).to(self.args.device)

        # perceiverIO
        self.perceiverIO = PerceiverIO(args)

        # Output optimal action
        if args.checkpoint_path == "":
            self.action_embedding = nn.Linear(args.emb, args.action_space_size + args.enemy_num)
        else:
            self.action_embedding = nn.Linear(args.emb, args.action_space_size + args.min_enemy_num)

    def init_hidden(self):
        return torch.zeros(1, self.args.emb).to(self.args.device)
    
    def save_query(self, path):
        torch.save(self.action_query, "{}/action_query.pt".format(path))

    def load_query(self, path):
        self.action_query = torch.load("{}/action_query.pt".format(path)).to(self.args.device)
        self.action_query.requires_grad =  False

    def forward(self, inputs, hidden_state, task_enemy_num = None, task_ally_num = None, env = "sc2"):
        
        b, t, e = inputs.size()

        query = torch.repeat_interleave(torch.unsqueeze(self.action_query, dim = 0), b, dim = 0)
        outputs, hidden = self.perceiverIO(inputs, hidden_state, query)

        # first output for 6 action (no_op stop up down left right)
        q = self.action_embedding(outputs.view(-1, self.args.emb)).view(b, -1, 1)
        
        # last dim for hidden state
        h = hidden[:, -1:, :]

        return q, h

    def fixed_models_weight(self):
        self.action_embedding = nn.Linear(self.args.emb, self.args.action_space_size + self.args.enemy_num)
        self.perceiverIO.requires_grad = False
        