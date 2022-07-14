
smac_maps_features = {
    "3m":           [4, 5, 5, 1],      # feature size of action_move/enemy/ally/own
    "8m":           [4, 5, 5, 1],
    "2s3z":         [4, 8, 8, 4],
    "3s_vs_3z":     [4, 6, 6, 2], 
    "3s_vs_4z":     [4, 6, 6, 2], 
    "3s5z_vs_3s6z": [4, 8, 8, 4],
    "6h_vs_8z":     [4, 6, 5, 1], 
    "25m":          [4, 5, 5, 1],
    "bane_vs_bane": [4, 7, 7, 3], 
    "corridor":     [4, 5, 6, 2],
    "2c_vs_64zg":   [4, 5, 6, 2], 
    "5m_vs_6m":     [4, 5, 5, 1],
    "8m_vs_9m":     [4, 5, 5, 1], 
    "10m_vs_11m":   [4, 5, 5, 1]
}

smac_maps_entities = {
    "3m":           [3, 3],      # # of agents/enemies
    "8m":           [8, 8],
    "2s3z":         [2, 3],
    "3s_vs_3z":     [3, 3], 
    "3s_vs_4z":     [3, 4], 
    "3s5z_vs_3s6z": [8, 9],
    "6h_vs_8z":     [6, 8], 
    "25m":          [25, 25],
    "bane_vs_bane": [24, 24], 
    "corridor":     [6, 24],
    "2c_vs_64zg":   [2, 64], 
    "5m_vs_6m":     [5, 6],
    "8m_vs_9m":     [8, 9], 
    "10m_vs_11m":   [10, 11]
}

def get_smac_map_config(env_name):

    assert smac_maps_entities[env_name] is not None

    return {
        "ally_num": smac_maps_entities[env_name][0], 
        "enemy_num": smac_maps_entities[env_name][1],
        "enemy_feature": smac_maps_features[env_name][1],
        "own_feature": smac_maps_features[env_name][3],
        "token_dim": smac_maps_features[env_name][0] + smac_maps_features[env_name][3],
        "env_args": {
            "map_name": env_name
        }
    }
