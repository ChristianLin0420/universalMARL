
from os import stat
import numpy as np
import gym

from gym import spaces
from gym.spaces import prng
from .multiagentenv import MultiAgentEnv
from .multiparticleenv.multi_discrete import MultiDiscrete


class MultiParticleEnv(MultiAgentEnv):
    """The Multi-agent Particle environment"""

    def __init__(
        self, 
        world, 
        reset_callback = None, 
        reward_callback = None,
        observation_callback = None, 
        info_callback = None,
        done_callback = None, 
        shared_viewer = True, 
        seed = None):

        self.viewer_width = 400.0
        self.viewer_height = 400.0

        self.world = world
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # environment parameters
        self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = True
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0

        self._seed = seed
        self.episode_limit = 30
        self._episode_count = 0
        self._episode_steps = 0

        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            total_action_space = []
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
            else:
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,), dtype=np.float32)
            if agent.movable:
                total_action_space.append(u_action_space)
            # communication action space
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c)
            else:
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)
            if not agent.silent:
                total_action_space.append(c_action_space)
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            agent.action.c = np.zeros(self.world.dim_c)

        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def step(self, actions):
        obs_n = []
        reward_n = []
        info_n = {}
        self.agents = self.world.policy_agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(actions[i], agent, self.action_space[i])

        # advance world state
        self.world.step()
        self._episode_steps += 1

        # record observation for each agent
        for i, agent in enumerate(self.agents):
            obs_n.append(self.get_obs_agent(i))
            reward_n.append(self._get_reward(agent))
            # info_n['n'].append(self._get_info(agent))

        done = self._get_done()

        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)

        if self._episode_steps >= self.episode_limit:
            done = True

        if done:
            self._episode_count += 1

        return reward, done, info_n

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index+s)])
                index += s
            action = act
        else:
            action = [action]

        if agent.movable:
            # physical action
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
                # process discrete action
                if action[0] == 1: agent.action.u[0] = -1.0
                if action[0] == 2: agent.action.u[0] = +1.0
                if action[0] == 3: agent.action.u[1] = -1.0
                if action[0] == 4: agent.action.u[1] = +1.0
            else:
                if self.force_discrete_action:
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:
                    agent.action.u[0] += action[0][1] - action[0][2]
                    agent.action.u[1] += action[0][3] - action[0][4]
                else:
                    agent.action.u = action[0]
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        if not agent.silent:
            # communication action
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self):
        if self.done_callback is None:
            return False
        return self.done_callback(self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    def get_obs(self):
        agents_obs = [self.get_obs_agent(i) for i in range(self.n)]
        return agents_obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(self.world.agents[agent_id], self.world)

    def get_obs_size(self):
        """ Returns the shape of the observation """
        agents_obs_size = len(self.world.agents) * (self.world.dim_p * 2 + self.world.dim_c)
        return agents_obs_size

    def get_state(self):
        """Returns the global state.
        NOTE: This functon should not be used during decentralised execution.
        """

        ally_state = []
        enemy_state = []

        for agent in self.world.agents:
            if agent.adversary:
                enemy_state.append(self.observation_callback(agent, self.world))
            else:
                ally_state.append(self.observation_callback(agent, self.world))

        ally_state = np.asarray(ally_state)
        enemy_state = np.asarray(enemy_state)

        state = np.append(ally_state.flatten(), enemy_state.flatten())
        state = state.astype(dtype = np.float32)

        return state

    def get_state_size(self):
        size = 0

        for i in range(len(self.world.agents)):
            size += len(self.observation_callback(self.world.agents[i], self.world))

        return size

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(len(self.agents)):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def check_bounds(self, x, y):
        """Whether a point is within the map bounds."""
        return 0 <= x < self.viewer_width and 0 <= y < self.viewer_height

    def can_move(self, agent, direction):
        m = 1.0

        if direction == 1:
            x, y = int(agent.state.p_pos[0]), int(agent.state.p_pos[1] - m)
        elif direction == 2:
            x, y = int(agent.state.p_pos[0] + m), int(agent.state.p_pos[1])
        elif direction == 3:
            x, y = int(agent.state.p_pos[0]), int(agent.state.p_pos[1] + m)
        else:
            x, y = int(agent.state.p_pos[0] - m), int(agent.state.p_pos[1])

        if self.check_bounds(x, y):
            return True

        return False

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        
        agent = self.agents[agent_id]

        avail_actions = [0] * 5
        avail_actions[0] = 1

        # see if we can move
        if self.can_move(agent, 1):
            avail_actions[1] = 1
        if self.can_move(agent, 2):
            avail_actions[2] = 1
        if self.can_move(agent, 3):
            avail_actions[3] = 1
        if self.can_move(agent, 4):
            avail_actions[4] = 1

        return avail_actions

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return 5

    def reset(self):
        # reset world
        self.reset_callback(self.world)

        # reset renderer
        self._reset_render()

        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for id in range(len(self.agents)):
            obs_n.append(self.get_obs_agent(id))

        return obs_n

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    def render(self, mode='human'):
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent: continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            # print(message)

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                #from gym.envs.classic_control import rendering
                from .multiparticleenv import rendering
                self.viewers[i] = rendering.Viewer(self.viewer_width, self.viewer_height)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            #from gym.envs.classic_control import rendering
            from .multiparticleenv import rendering
            self.render_geoms = []
            self.render_geoms_xform = []

            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []

        for i in range(len(self.viewers)):
            from .multiparticleenv import rendering
            # update bounds to center around agent
            cam_range = 1
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array = mode=='rgb_array'))

        return results

    def close(self):
        self.render_geoms = None
        self.render_geoms_xform = None
        self.viewers = None

    def seed(self):
        return self._seed

    def save_replay(self):
        ''' Do nothing '''

    def get_stats(self):
        stats = {
            # "battles_won": self.battles_won,
            # "battles_game": self.battles_game,
            # "battles_draw": self.timeouts,
            # "win_rate": self.battles_won / self.battles_game,
            # "timeouts": self.timeouts,
            # "restarts": self.force_restarts,
        }
        return stats

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n,
                    "episode_limit": self.episode_limit}
        return env_info


class MultiDiscrete(gym.Space):
    """
    - The multi-discrete action space consists of a series of discrete action spaces with different parameters
    - It can be adapted to both a Discrete action space or a continuous (Box) action space
    - It is useful to represent game controllers or keyboards where each key can be represented as a discrete action space
    - It is parametrized by passing an array of arrays containing [min, max] for each discrete action space
       where the discrete action space can take any integers from `min` to `max` (both inclusive)
    Note: A value of 0 always need to represent the NOOP action.
    e.g. Nintendo Game Controller
    - Can be conceptualized as 3 discrete action spaces:
        1) Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: min: 0, max: 4
        2) Button A:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
        3) Button B:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
    - Can be initialized as
        MultiDiscrete([ [0,4], [0,1], [0,1] ])
    """
    def __init__(self, array_of_param_array):
        self.low = np.array([x[0] for x in array_of_param_array])
        self.high = np.array([x[1] for x in array_of_param_array])
        self.num_discrete_space = self.low.shape[0]

    def sample(self):
        """ Returns a array with one sample from each discrete action space """
        # For each row: round(random .* (max - min) + min, 0)
        random_array = prng.np_random.rand(self.num_discrete_space)
        return [int(x) for x in np.floor(np.multiply((self.high - self.low + 1.), random_array) + self.low)]
    def contains(self, x):
        return len(x) == self.num_discrete_space and (np.array(x) >= self.low).all() and (np.array(x) <= self.high).all()

    @property
    def shape(self):
        return self.num_discrete_space
    def __repr__(self):
        return "MultiDiscrete" + str(self.num_discrete_space)
    def __eq__(self, other):
        return np.array_equal(self.low, other.low) and np.array_equal(self.high, other.high)