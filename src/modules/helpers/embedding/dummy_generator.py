import torch

class DummyGenerator():

    def __init__(self, device):
        super(DummyGenerator, self).__init__()

        self.device = device

    def generateRandomEntity(self, batch, length, feature):
        dummy = torch.zeros(batch, length, feature, device=self.device)

        # visible
        visible = torch.zeros(batch, length, 1)

        # other features
        damage = torch.randint(0, 1, (batch, length, 4))
        features = torch.rand(batch, length, 4)
        features = torch.mul(features, damage)

        dummy[:, :, :1] = visible
        dummy[:, :, 1:] = features

        return dummy

    def generateAverageEntity(self, entity, length):

        # visible
        visible = torch.zeros(entity.size(0), length, 1)
        
        mean_entity = torch.mean(entity, 1, True)
        dummy = torch.repeat_interleave(mean_entity, length, 1)

        dummy[:, :, :1] = visible

        return dummy


