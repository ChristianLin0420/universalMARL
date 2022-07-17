import torch
import torch.nn as nn

from torch.nn import functional as F
from modules.helpers.layers.transformer_decoder_layer import DecoderLayer

class madtGPT(nn.Module):

    def __init__(self, args, mask, drop_prob):
        super().__init__()

        self.args = args

        self.model_type = args.model_type
        self.state_size = args.state_size

        # input embedding stem
        self.tok_emb = nn.Embedding(args.vocab_size, args.n_embd)
        # self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.pos_emb = nn.Parameter(torch.zeros(1, args.block_size + 1, args.emb))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, args.max_timestep + 1, args.emb))
        self.drop = nn.Dropout(0.)


        self.layers = nn.ModuleList([DecoderLayer(emb=args.emb,
                                                  heads=args.heads,
                                                  mask=True,
                                                  dropout=drop_prob)
                                     for _ in range(args.depth)])

        # decoder head
        self.ln_f = nn.LayerNorm(args.emb)
        if self.model_type == 'actor':
            self.head = nn.Linear(args.emb, args.vocab_size, bias=False)
        elif self.model_type == 'critic':
            self.head = nn.Linear(args.emb, 1, bias=False)
        else:
            raise NotImplementedError

        self.block_size = args.block_size
        self.apply(self._init_weights)

        self.state_encoder = nn.Sequential(nn.Linear(self.state_size, args.emb), nn.Tanh())

        self.ret_emb = nn.Sequential(nn.Linear(1, args.emb), nn.Tanh())

        self.mask_emb = nn.Sequential(nn.Linear(1, args.emb), nn.Tanh())

        self.action_embeddings = nn.Sequential(nn.Embedding(args.vocab_size, args.emb), nn.Tanh())
        nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config, lr):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        no_decay.add('global_pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=train_config.betas)

        return optimizer

    def forward(self, states, pre_actions, rtgs = None, timesteps = None):
        # states: (batch, context_length, 4*84*84)
        # actions: (batch, context_length, 1)
        # targets: (batch, context_length, 1)
        # rtgs: (batch, context_length, 1)
        # timesteps: (batch, context_length, 1)

        state_embeddings = self.state_encoder(
            states.reshape(-1, self.state_size).type(torch.float32).contiguous())
        state_embeddings = state_embeddings.reshape(states.shape[0], states.shape[1],
                                                    self.config.n_embd)  # (batch, block_size, n_embd)

        if self.model_type == 'rtgs_state_action':
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))

            action_embeddings = self.action_embeddings(
                pre_actions.type(torch.long).squeeze(-1))  # (batch, block_size, n_embd)

            token_embeddings = torch.zeros(
                (states.shape[0], states.shape[1] * 3, self.config.n_embd), dtype=torch.float32,
                device=state_embeddings.device)
            token_embeddings[:, ::3, :] = rtg_embeddings
            token_embeddings[:, 1::3, :] = state_embeddings
            token_embeddings[:, 2::3, :] = action_embeddings
            num_elements = 3
        elif self.model_type == 'state_action':
            action_embeddings = self.action_embeddings(
                pre_actions.type(torch.long).squeeze(-1))  # (batch, block_size, n_embd)

            token_embeddings = torch.zeros(
                (states.shape[0], states.shape[1] * 2, self.config.n_embd), dtype=torch.float32,
                device=state_embeddings.device)
            token_embeddings[:, ::2, :] = state_embeddings
            token_embeddings[:, 1::2, :] = action_embeddings
            num_elements = 2
        elif self.model_type == 'state_only':
            token_embeddings = state_embeddings
            num_elements = 1
        else:
            raise NotImplementedError()

        batch_size = states.shape[0]
        all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size, dim=0)
        global_pos_emb = torch.gather(all_global_pos_emb, 1, torch.repeat_interleave(timesteps, self.config.n_embd, dim=-1))
        global_pos_emb = torch.repeat_interleave(global_pos_emb, num_elements, dim=1)
        context_pos_emb = self.pos_emb[:, :token_embeddings.shape[1], :]
        position_embeddings = global_pos_emb + context_pos_emb

        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        if self.model_type == 'rtgs_state_action':
            # logits = logits[:, 1::3, :]  # only keep predictions from state_embeddings
            logits = logits[:, 2::3, :]  # consider all tokens
        elif self.model_type == 'state_action':
            # logits = logits[:, ::2, :]  # only keep predictions from state_embeddings
            logits = logits[:, 1::2, :]  # consider all tokens
        elif self.model_type == 'state_only':
            logits = logits
        else:
            raise NotImplementedError()

        return logits