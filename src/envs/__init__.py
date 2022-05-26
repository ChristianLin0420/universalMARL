from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
from .multiparticlesenv import MultiParticleEnv
import envs.multiparticleenv.scenarios as scenarios
import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

### particle environment setting
# load scenario from script
def get_particle_env(env, **kwargs) -> MultiAgentEnv:
    scenario = scenarios.load(kwargs["scenario"] + ".py").Scenario()
    # create world
    world = scenario.make_world(kwargs["n_agents"], kwargs["n_landmarks"])
    return env(world, scenario.reset_world, scenario.reward, scenario.observation, None, scenario.game_over, kwargs["share_view"], kwargs["seed"])

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["particle"] = partial(get_particle_env, env=MultiParticleEnv)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
