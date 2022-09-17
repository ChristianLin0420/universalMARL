import torch.nn as nn
import torch

from modules.helpers.models.perceiverIO import PerceiverIO


class PerceiverIOplusAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(PerceiverIOplusAgent, self).__init__()

        self.args = args
        self.action_query = nn.Parameter(torch.rand(1, args.token_dim))

        # perceiverIO
        self.perceiverIO = PerceiverIO(args)

        # Output optimal action
        if args.checkpoint_path == "":
            self.action_embedding = nn.Linear(args.emb, args.action_space_size + args.enemy_num)
        else:
            self.action_embedding = nn.Linear(args.emb, args.action_space_size + args.min_enemy_num)

    def init_hidden(self):
        # make hidden states on same device as model
        if self.args.use_cuda:
            return torch.zeros(1, self.args.emb).cuda()
        else:
            return torch.zeros(1, self.args.emb)
    
    def save_query(self, path):
        torch.save(self.action_query, "{}/action_query.pt".format(path))

    def load_query(self, path):
        self.action_query = torch.load("{}/action_query.pt".format(path))

        if self.args.use_cuda:
            self.action_query = self.action_query.cuda()

        self.action_query.requires_grad =  False

    def forward(self, inputs, hidden_state, task_enemy_num = None, task_ally_num = None, env = "sc2"):
        
        b, t, e = inputs.size()

        inputs = torch.reshape(inputs, (b, t * e))
        new = torch.zeros(b, t * e)

        f_size = self.args.token_dim
        own_feature = 1
        move_feature = 4

        new[:, :move_feature] = inputs[:, :move_feature]                                # agent movement features
        new[:, move_feature:(move_feature + own_feature)] = inputs[:, -own_feature:]    # agent own feature
        new[:, f_size:(f_size + (task_ally_num - 1) * f_size)] = inputs[:, (move_feature + task_enemy_num * f_size):(t * e - own_feature)]  # ally features
        new[:, task_ally_num * f_size:] = inputs[:, move_feature:(move_feature + task_enemy_num * f_size)]                                  # enemy features
        new = torch.reshape(new, (b, t, e))

        if self.args.use_cuda:
            new = new.cuda()

        query = torch.repeat_interleave(torch.unsqueeze(self.action_query, dim = 0), b, dim = 0)

        outputs, hidden = self.perceiverIO(new, hidden_state, query)

        # first output for 6 action (no_op stop up down left right)
        q = self.action_embedding(outputs.view(-1, self.args.emb)).view(b, -1, 1)
        
        # last dim for hidden state
        h = hidden[:, -1:, :]

        return q, h

    def fixed_models_weight(self):
        self.action_embedding = nn.Linear(self.args.emb, self.args.action_space_size + self.args.enemy_num)
        self.perceiverIO.requires_grad = False
        