import torch

class DummyGenerator():

    def __init__(self, device):
        super(DummyGenerator, self).__init__()

        self.device = device

    def generateRandomEntity(self, batch, length, feature):
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

    def generateAverageEntity(self, entity, length):
        
        mean_entity = torch.mean(entity, 1, True)
        repeat = length - entity.size(1)
        dummy = torch.repeat_interleave(mean_entity, repeat, 1)

        return dummy


