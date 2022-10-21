import torch

class DummyGenerator():

    def __init__(self, device):
        super(DummyGenerator, self).__init__()

        self.device = device

    def generate(self, batch, length, feature):
        dummy = torch.zeros(batch, length, feature, device=self.device)

        # visible
        visible = torch.randint(0, 1, (batch, length, 1))

        # other features
        damage = torch.randint(0, 1, (batch, length, 4))
        features = torch.rand(batch, length, 4)
        features = torch.mul(features, damage)

        dummy[:, :, :1] = visible
        dummy[:, :, 1:] = features

        return dummy
