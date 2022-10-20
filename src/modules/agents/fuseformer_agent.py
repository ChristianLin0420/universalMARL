import torch.nn as nn
import torch
from modules.helpers.models.fuseformer import Fouseformer


class FouseformerAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(FouseformerAgent, self).__init__()
        self.args = args
        self.encoder_random_input = args.random_encoder_inputs_zero
        
        self.basic_action_query = torch.unsqueeze(torch.rand(args.action_space_size, args.emb), 0).to(self.args.device)
        self.transformer = Fouseformer(args, args.token_dim, args.emb)

        # Output optimal action
        self.action_embedding = nn.Sequential(
                                                nn.Linear(args.emb, args.emb),
                                                nn.ReLU(),
                                                nn.Linear(args.emb, 1)
                                            )

    def init_hidden(self):
        # make hidden states on same device as model
        return torch.zeros(1, self.args.emb).to(self.args.device)

    def forward(self, inputs, hidden_state, task_enemy_num = None, task_ally_num = None, env = "sc2"):
        
        b = inputs.size(0)

        encoder_inputs = inputs[:, :task_ally_num, :].to(self.args.device)
        decoder_inputs = inputs[:, task_ally_num:, :].to(self.args.device)

        outputs, hidden = self.transformer.forward(encoder_inputs, decoder_inputs, hidden_state, self.basic_action_query, None)

        q = self.action_embedding(outputs.contiguous().view(-1, self.args.emb)).view(b, -1, 1)

        return q, hidden

    def save_query(self, path):
        torch.save(self.basic_action_query, "{}/basic_action_query.pt".format(path))

    def load_query(self, path):
        self.basic_action_query = torch.load("{}/basic_action_query.pt".format(path)).to(self.args.device)
        self.basic_action_query.requires_grad =  False

    def fixed_models_weight(self):
        self.transformer.requires_grad = False
        self.action_embedding.requires_grad = True
