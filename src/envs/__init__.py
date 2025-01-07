from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)

### particle environment setting
# load scenario from script
# def get_particle_env(env, **kwargs) -> MultiAgentEnv:
#     scenario = scenarios.load(kwargs["scenario"] + ".py").Scenario()
#     # create world

#     if kwargs["scenario"] == "simple_spread": 
#         world = scenario.make_world(kwargs["n_agents"], kwargs["n_landmarks"])
#     elif kwargs["scenario"] == "simple_tag":
#         world = scenario.make_world(kwargs["n_agents"], kwargs["n_adverary"], kwargs["n_landmarks"])

#     return env(kwargs["scenario"], world, scenario.reset_world, scenario.reward, scenario.observation, None, scenario.game_over, kwargs["share_view"], kwargs["seed"])

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)

# particle environment
# REGISTRY["simple_spread"] = partial(get_particle_env, env=MultiParticleEnv)
# REGISTRY["simple_tag"] = partial(get_particle_env, env=MultiParticleEnv)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
