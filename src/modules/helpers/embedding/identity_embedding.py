import os
import json
import torch

from torch import nn

entity_role = {
    "marine": 0,
    "stalkers": 1,
    "zealots": 2,
    "colossus": 3,
    "medivac": 4,
    "marauder": 5,
    "spine_crawler": 6,
    "hydralisks": 7,
    "zerglines": 8,
    "banelings": 9
}

smac_maps_identity = {
    "3m": {
        "ally": [0] * 3,
        "enemy": [0] * 3
    }, 
    "8m": {
        "ally": [0] * 8,
        "enemy": [0] * 8
    }, 
    "2s3z": {
        "ally": [1, 1, 2, 2, 2],
        "enemy": [2, 2, 2, 1, 1]
    }, 
    "3s_vs_3z": {
        "ally": [1] * 3,
        "enemy": [2] * 3
    }, 
    "3s_vs_4z": {
        "ally": [1] * 3,
        "enemy": [2] * 4
    }, 
    "3s_vs_5z": {
        "ally": [1] * 3,
        "enemy": [2] * 5
    }, 
    "3s5z": {
        "ally": [1, 1, 1, 2, 2, 2, 2, 2],
        "enemy": [1, 1, 2, 2, 2, 2, 2, 1]
    }, 
    "3s5z_vs_3s6z": {
        "ally": [1, 1, 1, 2, 2, 2, 2, 2],
        "enemy": [1, 1, 2, 2, 2, 2, 2, 2, 1]
    }, 
    "6h_vs_8z": {
        "ally": [7] * 6,
        "enemy": [2] * 8
    }, 
    "25m": {
        "ally": [0] * 25,
        "enemy": [0] * 25
    }, 
    # "bane_vs_bane": {
    #     "ally": [0, 0, 0, 0, 0, 0, 0, 0],
    #     "enemy": [0, 0, 0, 0, 0, 0, 0, 0]
    # }, 
    "corridor": {
        "ally": [2] * 6,
        "enemy": [8] * 24
    }, 
    "2c_vs_64zg": {
        "ally": [3] * 2,
        "enemy": [8] * 64
    }, 
    "5m_vs_6m": {
        "ally": [0] * 5,
        "enemy": [0] * 6
    }, 
    "8m_vs_9m": {
        "ally": [0] * 8,
        "enemy": [0] * 9
    }, 
    "10m_vs_11m": {
        "ally": [0] * 10,
        "enemy": [0] * 11
    }, 
    "27m_vs_30m": {
        "ally": [0] * 27,
        "enemy": [0] * 30
    }
}


class IdentityEmbedding(nn.Module):
    def __init__(self, args, batch, emb, map_name, dummy, device = None):
        super(IdentityEmbedding, self).__init__()
        
        self.args = args
        self.batch = batch
        self.emb = emb
        self.map_name = map_name
        self.dummy = dummy
        self.device = device

        self.identity_embedding = None

        self.path = "src/modules/helpers/embedding/identity_{}.json".format(emb)

        self.generate_identity()

    def set_identity_embedding(self):
        
        if os.path.exists(self.path):
            with open(self.path) as json_file:
                self.id_emb = json.load(json_file)
        else:
            self.id_emb = {}

            for i in range(len(entity_role)):
                self.id_emb[str(i)] = torch.rand(self.emb).tolist()
            
            json_string = json.dumps(self.id_emb)

            with open(self.path, 'w') as outfile:
                outfile.write(json_string)


    def generate_identity(self):

        self.set_identity_embedding()
        
        ally_order = smac_maps_identity[self.map_name]["ally"]
        enemy_order = smac_maps_identity[self.map_name]["enemy"]

        for agent_idx, agent in enumerate(ally_order):
            agent_id_emb = torch.tensor([self.id_emb[str(agent)]])
            
            for ally_idx, ally in enumerate(ally_order):
                if agent_idx != ally_idx:
                    ally_id_emb = torch.tensor([self.id_emb[str(ally)]])
                    agent_id_emb = torch.cat((agent_id_emb, ally_id_emb), 0)

            if self.dummy:
                dummy_ally_identity_count = self.args.max_ally_num - self.args.ally_num
                for _ in range(dummy_ally_identity_count):
                    dummy_ally_id_emb = torch.tensor([[0] * self.emb])
                    agent_id_emb = torch.cat((agent_id_emb, dummy_ally_id_emb), 0)

            for enemy in enemy_order:
                enemy_id_emb = torch.tensor([self.id_emb[str(enemy)]])
                agent_id_emb = torch.cat((agent_id_emb, enemy_id_emb), 0)

            if self.dummy:
                dummy_enemy_identity_count = self.args.max_enemy_num - self.args.enemy_num
                for _ in range(dummy_enemy_identity_count):
                    dummy_enemy_id_emb = torch.tensor([[0] * self.emb])
                    agent_id_emb = torch.cat((agent_id_emb, dummy_enemy_id_emb), 0)

            if self.identity_embedding is None:
                self.identity_embedding = torch.unsqueeze(agent_id_emb, 0)
            else:
                self.identity_embedding = torch.cat((self.identity_embedding, torch.unsqueeze(agent_id_emb, 0)), 0)

        self.identity_embedding = self.identity_embedding.repeat(self.batch, 1, 1)
        self.identity_embedding.to(self.device)

    def get_identity_embedding(self):
        return self.identity_embedding
            





            
