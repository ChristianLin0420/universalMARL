


smac_maps_features = {
    "3m":           [4, 5, 5, 1, 3, 8, 8],      # feature size of action_move/enemy/ally/own/minE/maxE/maxA 
    "8m":           [4, 5, 5, 1, 3, 8, 8],
    "2s3z":         [4, 8, 8, 4, 5, 8, 8],
    "3s_vs_3z":     [4, 6, 6, 2, 3, 5, 3], # -> 3s_vs_5z
    "3s_vs_4z":     [4, 6, 6, 2, 4, 5, 3],
    "3s_vs_5z":     [4, 6, 6, 2, 3, 5, 3], 
    "3s5z_vs_3s6z": [4, 8, 8, 4, 9, 9, 8],
    "6h_vs_8z":     [4, 6, 5, 1, 8, 8, 6], 
    "25m":          [4, 5, 5, 1, 25, 25, 25],
    "bane_vs_bane": [4, 7, 7, 3, 9, 24, 24], 
    "corridor":     [4, 5, 6, 2, 9, 24, 6],
    "2c_vs_64zg":   [4, 5, 6, 2, 64, 64, 2], 
    "5m_vs_6m":     [4, 5, 5, 1, 6, 9, 8],
    "8m_vs_9m":     [4, 5, 5, 1, 6, 9, 8], 
    "10m_vs_11m":   [4, 5, 5, 1, 6, 11, 10],
    "27m_vs_30m":   [4, 5, 5, 1, 3, 30, 28]
}

smac_maps_entities = {
    "3m":           [3, 3],      # # of agents/enemies
    "8m":           [8, 8],
    "2s3z":         [5, 5],
    "3s_vs_3z":     [3, 3], 
    "3s_vs_4z":     [3, 4], 
    "3s_vs_5z":     [3, 5], 
    "3s5z_vs_3s6z": [8, 9],
    "6h_vs_8z":     [6, 8], 
    "25m":          [25, 25],
    "bane_vs_bane": [24, 24], 
    "corridor":     [6, 24],
    "2c_vs_64zg":   [2, 64], 
    "5m_vs_6m":     [5, 6],
    "8m_vs_9m":     [8, 9], 
    "10m_vs_11m":   [10, 11],
    "27m_vs_30m":   [27, 30]
}

''' 
Enemy information:

entity property: 
    self(0), ally(1), enemy(2)

entity role identification:
    marine: 0
    stalkers: 1
    zealots: 2
    colossus: 3
    medivac: 4
    marauder: 5
    spine crawler: 6
    hydralisks: 7
    zerglines: 8
    banelings: 9
'''

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

entity_identity = {
    "self": 0, 
    "ally": 1, 
    "enemy": 2
}

def get_smac_map_config(env_name):

    assert smac_maps_entities[env_name] is not None

    return {
        "ally_num": smac_maps_entities[env_name][0], 
        "enemy_num": smac_maps_entities[env_name][1],
        "enemy_feature":5, #smac_maps_features[env_name][1],
        "own_feature": 1, #smac_maps_features[env_name][3],
        "token_dim": 5, #smac_maps_features[env_name][0] + smac_maps_features[env_name][3],
        "min_enemy_num": smac_maps_features[env_name][4],
        "max_enemy_num": smac_maps_features[env_name][5],
        "max_ally_num": smac_maps_features[env_name][6],
        "env_args": {
            "map_name": env_name
        }
    }

def get_entity_extra_information(identity, role):
    return [entity_identity[identity], entity_role[role]]


from pysc2.maps import lib

class SMACMap(lib.Map):
    directory = "SMAC_Maps"
    download = "https://github.com/oxwhirl/smac#smac-maps"
    players = 2
    step_mul = 8
    game_steps_per_episode = 0


map_param_registry = {
    "3m": {
        "n_agents": 3,
        "n_enemies": 3,
        "limit": 60,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "8m": {
        "n_agents": 8,
        "n_enemies": 8,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "25m": {
        "n_agents": 25,
        "n_enemies": 25,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "5m_vs_6m": {
        "n_agents": 5,
        "n_enemies": 6,
        "limit": 70,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "8m_vs_9m": {
        "n_agents": 8,
        "n_enemies": 9,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "10m_vs_11m": {
        "n_agents": 10,
        "n_enemies": 11,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "27m_vs_30m": {
        "n_agents": 27,
        "n_enemies": 30,
        "limit": 180,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "MMM": {
        "n_agents": 10,
        "n_enemies": 10,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
    "MMM2": {
        "n_agents": 10,
        "n_enemies": 12,
        "limit": 180,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
    "2s3z": {
        "n_agents": 5,
        "n_enemies": 5,
        "limit": 120,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "3s5z": {
        "n_agents": 8,
        "n_enemies": 8,
        "limit": 150,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "3s5z_vs_3s6z": {
        "n_agents": 8,
        "n_enemies": 9,
        "limit": 170,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "3s_vs_3z": {
        "n_agents": 3,
        "n_enemies": 3,
        "limit": 150,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 0,
        "map_type": "stalkers",
    },
    "3s_vs_4z": {
        "n_agents": 3,
        "n_enemies": 4,
        "limit": 200,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 0,
        "map_type": "stalkers",
    },
    "3s_vs_5z": {
        "n_agents": 3,
        "n_enemies": 5,
        "limit": 250,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 0,
        "map_type": "stalkers",
    },
    "1c3s5z": {
        "n_agents": 9,
        "n_enemies": 9,
        "limit": 180,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 3,
        "map_type": "colossi_stalkers_zealots",
    },
    "2m_vs_1z": {
        "n_agents": 2,
        "n_enemies": 1,
        "limit": 150,
        "a_race": "T",
        "b_race": "P",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "corridor": {
        "n_agents": 6,
        "n_enemies": 24,
        "limit": 400,
        "a_race": "P",
        "b_race": "Z",
        "unit_type_bits": 0,
        "map_type": "zealots",
    },
    "6h_vs_8z": {
        "n_agents": 6,
        "n_enemies": 8,
        "limit": 150,
        "a_race": "Z",
        "b_race": "P",
        "unit_type_bits": 0,
        "map_type": "hydralisks",
    },
    "2s_vs_1sc": {
        "n_agents": 2,
        "n_enemies": 1,
        "limit": 300,
        "a_race": "P",
        "b_race": "Z",
        "unit_type_bits": 0,
        "map_type": "stalkers",
    },
    "so_many_baneling": {
        "n_agents": 7,
        "n_enemies": 32,
        "limit": 100,
        "a_race": "P",
        "b_race": "Z",
        "unit_type_bits": 0,
        "map_type": "zealots",
    },
    "bane_vs_bane": {
        "n_agents": 24,
        "n_enemies": 24,
        "limit": 200,
        "a_race": "Z",
        "b_race": "Z",
        "unit_type_bits": 2,
        "map_type": "bane",
    },
    "2c_vs_64zg": {
        "n_agents": 2,
        "n_enemies": 64,
        "limit": 400,
        "a_race": "P",
        "b_race": "Z",
        "unit_type_bits": 0,
        "map_type": "colossus",
    },

    # This is adhoc environment
    "1c2z_vs_1c1s1z": {
        "n_agents": 3,
        "n_enemies": 3,
        "limit": 180,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 3,
        "map_type": "colossi_stalkers_zealots",
    },
    "1c2s_vs_1c1s1z": {
        "n_agents": 3,
        "n_enemies": 3,
        "limit": 180,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 3,
        "map_type": "colossi_stalkers_zealots",
    },
    "2c1z_vs_1c1s1z": {
        "n_agents": 3,
        "n_enemies": 3,
        "limit": 180,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 3,
        "map_type": "colossi_stalkers_zealots",
    },
    "2c1s_vs_1c1s1z": {
        "n_agents": 3,
        "n_enemies": 3,
        "limit": 180,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 3,
        "map_type": "colossi_stalkers_zealots",
    },
    "1c1s1z_vs_1c1s1z": {
        "n_agents": 3,
        "n_enemies": 3,
        "limit": 180,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 3,
        "map_type": "colossi_stalkers_zealots",
    },

    "3s5z_vs_4s4z": {
        "n_agents": 8,
        "n_enemies": 8,
        "limit": 150,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "4s4z_vs_4s4z": {
        "n_agents": 8,
        "n_enemies": 8,
        "limit": 150,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "5s3z_vs_4s4z": {
        "n_agents": 8,
        "n_enemies": 8,
        "limit": 150,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "6s2z_vs_4s4z": {
        "n_agents": 8,
        "n_enemies": 8,
        "limit": 150,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "2s6z_vs_4s4z": {
        "n_agents": 8,
        "n_enemies": 8,
        "limit": 150,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },

    "6m_vs_6m_tz": {
        "n_agents": 6,
        "n_enemies": 6,
        "limit": 70,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "5m_vs_6m_tz": {
        "n_agents": 5,
        "n_enemies": 6,
        "limit": 70,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "3s6z_vs_3s6z": {
        "n_agents": 9,
        "n_enemies": 9,
        "limit": 170,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "7h_vs_8z": {
        "n_agents": 7,
        "n_enemies": 8,
        "limit": 150,
        "a_race": "Z",
        "b_race": "P",
        "unit_type_bits": 0,
        "map_type": "hydralisks",
    },
    "2s2z_vs_zg": {
        "n_agents": 4,
        "n_enemies": 20,
        "limit": 200,
        "a_race": "P",
        "b_race": "Z",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots_vs_zergling",
    },
    "1s3z_vs_zg": {
        "n_agents": 4,
        "n_enemies": 20,
        "limit": 200,
        "a_race": "P",
        "b_race": "Z",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots_vs_zergling",
    },
    "3s1z_vs_zg": {
        "n_agents": 4,
        "n_enemies": 20,
        "limit": 200,
        "a_race": "P",
        "b_race": "Z",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots_vs_zergling",
    },

    "2s2z_vs_zg_easy": {
        "n_agents": 4,
        "n_enemies": 18,
        "limit": 200,
        "a_race": "P",
        "b_race": "Z",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots_vs_zergling",
    },
    "1s3z_vs_zg_easy": {
        "n_agents": 4,
        "n_enemies": 18,
        "limit": 200,
        "a_race": "P",
        "b_race": "Z",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots_vs_zergling",
    },
    "3s1z_vs_zg_easy": {
        "n_agents": 4,
        "n_enemies": 18,
        "limit": 200,
        "a_race": "P",
        "b_race": "Z",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots_vs_zergling",
    },
    "28m_vs_30m": {
        "n_agents": 28,
        "n_enemies": 30,
        "limit": 180,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "29m_vs_30m": {
        "n_agents": 29,
        "n_enemies": 30,
        "limit": 180,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "30m_vs_30m": {
        "n_agents": 30,
        "n_enemies": 30,
        "limit": 180,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "MMM2_test": {
        "n_agents": 10,
        "n_enemies": 12,
        "limit": 180,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
}


def get_smac_map_registry():
    return map_param_registry


for name in map_param_registry.keys():
    globals()[name] = type(name, (SMACMap,), dict(filename=name))


def get_map_params(map_name):
    map_param_registry = get_smac_map_registry()
    return map_param_registry[map_name]
