from modules.agents import REGISTRY as agent_REGISTRY, TRANSFORMERbasedAgent
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


# This multi-agent controller shares parameters between agents
class MADTMAC:
    def __init__(self, args):
        self.args = args
        self._build_agents()

    def forward(self, o, pre_a, rtg, t):
        return self.actor(o, pre_a, rtg, t)

    def parameters(self):
        return self.actor.parameters()

    def load_state(self, other_mac):
        self.actor.load_state_dict(other_mac.actor.state_dict())
        self.critic.load_state_dict(other_mac.critic.state_dict())

    def cuda(self):
        self.actor.cuda()
        self.critic.cuda()

    def save_models(self, path):
        th.save(self.actor.state_dict(), "{}/actor.th".format(path))
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))

    def load_models(self, path):
        self.actor.load_state_dict(th.load("{}/actor.th".format(path), map_location=lambda storage, loc: storage))
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self):
        self.actor = agent_REGISTRY[self.args.agent](self.args, "actor")
        self.critic = agent_REGISTRY[self.args.agent](self.args, "critic")