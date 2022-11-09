from logging import error
import torch.nn as nn
import torch
from modules.helpers.models.fuseformer_plus import FouseformerPlus
from modules.helpers.embedding.dummy_generator import DummyGenerator


class FouseformerPlusAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(FouseformerPlusAgent, self).__init__()
        self.args = args

        self.dummy_generator = DummyGenerator(args.device)
        
        self.transformer = FouseformerPlus(args, args.token_dim, args.emb)

        # Output optimal action
        if args.checkpoint_path == "":
            self.basic_action_embedding = nn.Linear(args.emb, args.action_space_size + args.enemy_num)
        else:
            self.basic_action_embedding = nn.Linear(args.emb, args.action_space_size + args.min_enemy_num)

    def init_hidden(self):
        # make hidden states on same device as model
        return torch.zeros(1, self.args.emb).to(self.args.device)

    def forward(self, inputs, hidden_state, decoder_outs, task_enemy_num = None, task_ally_num = None, env = "sc2"):
        
        b = inputs.size(0)

        encoder_inputs = inputs[:, :task_ally_num, :].to(self.args.device)
        decoder_inputs = inputs[:, task_ally_num:, :].to(self.args.device)

        if self.args.dummy_entity:
            encoder_dummy_length = self.args.max_ally_num - task_ally_num
            decoder_dummy_length = self.args.max_enemy_num - task_enemy_num

            if self.args.dummy_type == 1:
                encoder_dummy = self.dummy_generator.generateRandomEntity(b, encoder_dummy_length, self.args.token_dim)
                decoder_dummy = self.dummy_generator.generateRandomEntity(b, decoder_dummy_length, self.args.token_dim)
            elif self.args.dummy_type == 2:
                encoder_dummy = self.dummy_generator.generateAverageEntity(encoder_inputs, encoder_dummy_length)
                decoder_dummy = self.dummy_generator.generateAverageEntity(decoder_inputs, decoder_dummy_length)
            else:
                error("Given dummy type is not available.")

            encoder_inputs = torch.cat((encoder_inputs, encoder_dummy), 1)
            decoder_inputs = torch.cat((decoder_inputs, decoder_dummy), 1)

        decoder_outs, hidden = self.transformer.forward(encoder_inputs, decoder_inputs, hidden_state, decoder_outs, None)
        q = self.basic_action_embedding(decoder_outs[:, :1, :].view(-1, self.args.emb)).view(b, -1, 1)

        return q, hidden, decoder_outs[:, :1, :]

    def fixed_models_weight(self):
        self.basic_action_embedding = nn.Linear(self.args.emb, self.args.action_space_size + self.args.enemy_num)
        self.transformer.requires_grad = False
        self.basic_action_embedding.requires_grad = True
