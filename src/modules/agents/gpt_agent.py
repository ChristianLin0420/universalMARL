import torch.nn as nn
import torch
from modules.helpers.models.gpt import GPT


class GPTAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(GPTAgent, self).__init__()

        self.args = args

        self.transformer = GPT(args, args.token_dim, args.emb)

        # action parameters
        self.action_query = torch.unsqueeze(torch.rand(args.action_space_size, args.token_dim), 0)
        self.action_embedding = nn.Linear(self.args.emb, 1)

    def init_hidden(self):
        # make hidden states on same device as model
        if self.args.use_cuda:
            return torch.zeros(1, self.args.emb).cuda()
        else:
            return torch.zeros(1, self.args.emb)

    def forward(self, inputs, hidden_state, task_enemy_num = None, task_ally_num = None, env = "sc2"):

        b, t, e = inputs.size()

        encoder_inputs = inputs[:, :task_ally_num, :].to(self.args.device)
        decoder_inputs = inputs[:, task_ally_num:, :]
        decoder_inputs = torch.cat((encoder_inputs[:, :1, :], decoder_inputs), dim = 1)
        decoder_inputs = torch.cat((torch.repeat_interleave(self.action_query, b, 0), decoder_inputs), 1).to(self.args.device)

        outputs, tokens = self.transformer.forward(encoder_inputs, decoder_inputs, hidden_state, None)
        # first output for 6 action (no_op stop up down left right)
        q = self.action_embedding(outputs.view(-1, self.args.emb)).view(b, -1, 1)
        # last dim for hidden state
        h = outputs[:, -1:, :]

        return q[:, :-1, :], h

    def fixed_models_weight(self):
        self.transformer.requires_grad =  False
        self.action_embedding.requires_grad = True