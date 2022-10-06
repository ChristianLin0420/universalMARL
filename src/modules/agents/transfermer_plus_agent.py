import torch.nn as nn
import torch
from modules.helpers.models.transfer_transformer_plus import TransferableTransformerPlus


class TransfermerPlus(nn.Module):
    def __init__(self, input_shape, args):
        super(TransfermerPlus, self).__init__()
        self.args = args
        self.encoder_random_input = args.random_encoder_inputs_zero
        
        self.basic_action_query = torch.unsqueeze(torch.rand(args.action_space_size, args.token_dim), 0).to(self.args.device)
        self.transformer = TransferableTransformerPlus(args, args.token_dim, args.emb)

        # Output optimal action
        self.action_embedding = nn.Linear(args.emb, 1)

    def init_hidden(self):
        # make hidden states on same device as model
        return torch.zeros(1, self.args.emb).to(self.args.device)

    def forward(self, inputs, hidden_state, task_enemy_num = None, task_ally_num = None, env = "sc2"):
        
        b = inputs.size(0)

        encoder_inputs = inputs[:, :task_ally_num, :].to(self.args.device)
        decoder_inputs = inputs[:, task_ally_num:, :].to(self.args.device)
        decoder_inputs = torch.cat((torch.repeat_interleave(self.basic_action_query, b, 0), decoder_inputs), 1)

        # dummy input with random order
        if self.encoder_random_input and self.args.max_ally_num - task_ally_num > 0 :
            dummy_size = self.args.max_ally_num - task_ally_num
            dummy_inputs = torch.rand(b, dummy_size, inputs.size(-1)).to(self.args.device)
            dummy_inputs = torch.cat((encoder_inputs[:, 1:, :], dummy_inputs), 1)

            if self.args.random_inputs:
                permutation = torch.randperm(self.args.max_ally_num - 1).to(self.args.device)
                dummy_inputs = dummy_inputs[:, permutation, :]
            
            encoder_inputs = torch.cat((encoder_inputs[:, :1, :], dummy_inputs), 1)

        outputs = self.transformer.forward(encoder_inputs, decoder_inputs, hidden_state, None)

        q = self.action_embedding(outputs[:, :-1, :].contiguous().view(-1, self.args.emb)).view(b, -1, 1)

        # last dim for hidden state
        h = outputs[:, -1:, :]

        return q, h

    def save_query(self, path):
        torch.save(self.basic_action_query, "{}/basic_action_query.pt".format(path))

    def load_query(self, path):
        self.basic_action_query = torch.load("{}/basic_action_query.pt".format(path)).to(self.args.device)
        self.basic_action_query.requires_grad =  False

    def fixed_models_weight(self):
        self.transformer.requires_grad = False
        self.action_embedding.requires_grad = True
