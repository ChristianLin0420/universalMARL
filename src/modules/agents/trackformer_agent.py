import torch.nn as nn
import torch
from modules.helpers.models.track_transformer import TrackFormer


class TrackformerAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(TrackformerAgent, self).__init__()

        self.args = args
        self.track_emb = args.emb // 2

        self.transformer = TrackFormer(args, args.token_dim, args.emb)

        # action parameters
        self.action_query = torch.unsqueeze(torch.rand(args.action_space_size, args.token_dim), 0)
        self.action_embedding = nn.Linear(self.track_emb, 1)

    def init_hidden(self):
        # make hidden states on same device as model
        if self.args.use_cuda:
            return torch.zeros(1, self.track_emb).cuda()
        else:
            return torch.zeros(1, self.track_emb)

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
        encoder_inputs = new[:, :task_ally_num, :]
        decoder_inputs = new[:, task_ally_num:, :]
        decoder_inputs = torch.cat((torch.repeat_interleave(self.action_query, b, 0), decoder_inputs), 1)

        if self.args.use_cuda:
            encoder_inputs = encoder_inputs.cuda()
            decoder_inputs = decoder_inputs.cuda()

        outputs, tokens = self.transformer.forward(encoder_inputs, decoder_inputs, hidden_state, None)
        # first output for 6 action (no_op stop up down left right)
        q = self.action_embedding(outputs.view(-1, self.track_emb)).view(b, -1, 1)
        # last dim for hidden state
        h = tokens[:, -1:, :]

        return q, h

    def fixed_models_weight(self):
        self.transformer.requires_grad =  False
        self.action_embedding.requires_grad = True