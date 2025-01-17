import torch.nn as nn
import torch

from modules.helpers.models.perceiverIO import PerceiverIO


class PerceiverIOAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(PerceiverIOAgent, self).__init__()

        self.args = args
        self.action_query = nn.Parameter(torch.rand(args.action_space_size, args.token_dim))

        # perceiverIO
        self.perceiverIO = PerceiverIO(args)

        # Output optimal action
        self.action_embedding = nn.Linear(args.emb, 1)

    def init_hidden(self):
        # make hidden states on same device as model
        return torch.zeros(1, self.args.emb).to(self.args.device)
    
    def save_query(self, path):
        torch.save(self.action_query, "{}/action_query.pt".format(path))

    def load_query(self, path):
        self.action_query = torch.load("{}/action_query.pt".format(path)).to(self.args.device)
        self.action_query.requires_grad =  False

    def forward(self, inputs, hidden_state, task_enemy_num = None, task_ally_num = None, env = "sc2"):
        
        b, t, e = inputs.size()

        encoder_inputs = inputs[:, :task_ally_num, :].to(self.args.device)
        decoder_inputs = inputs[:, task_ally_num:, :]
        query = torch.repeat_interleave(torch.unsqueeze(self.action_query, dim = 0), b, dim = 0)
        decoder_inputs = torch.cat([query, decoder_inputs], dim = 1).to(self.args.device)

        outputs, hidden = self.perceiverIO(encoder_inputs, hidden_state, decoder_inputs)

        # first output for 6 action (no_op stop up down left right)
        q = self.action_embedding(outputs.view(-1, self.args.emb)).view(b, -1, 1)
        
        # last dim for hidden state
        h = hidden[:, -1:, :]

        return q, h

    def fixed_models_weight(self):
        self.perceiverIO.requires_grad = False
        self.action_embedding.requires_grad = True
        