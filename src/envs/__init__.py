from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
from .multiparticlesenv import MultiParticleEnv
from .multiparticlesenv import scenarios
import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

### particle environment setting
# load scenario from script
def get_particle_env(env, **kwargs) -> MultiAgentEnv:
    scenario = scenarios.load(**kwargs.env_args.senario + ".py").Scenario()
    # create world
    world = scenario.make_world(n_agents = **kwargs.n_agents, n_landmarks = **kwargs.n_landmarks)
    return env(world, world.reset_world, world.reward, world.observation, None, world.game_over, False, **kwargs.seed)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["particle"] = partial(get_particle_env, env=MultiParticleEnv)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
