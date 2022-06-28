
import os
import json
import torch
from torch import nn

class RandomLayer(nn.Module):
    def __init__(self, args):
        super(RandomLayer, self).__init__()

        self.args = args
        self.path = "src/modules/helpers/embedding/random_layer.json"

    def get_random_vector(self, b, t, e):

        if os.path.exists(self.path):
            with open(self.path) as json_file:
                data = json.load(json_file)["vectors"]
                data = torch.tensor(data)
                data = data[:b, :t, :e]

                if self.args.use_cuda:
                    return data.cuda()
                else:
                    return data
        else:
            tmp = torch.rand(64, 50, 64)
            data = tmp.tolist()

            json_string = {"vectors" : data}
            json_string = json.dumps(json_string)

            with open(self.path, 'w') as outfile:
                outfile.write(json_string)

            if self.args.use_cuda:
                return tmp[:b, :t, :e].cuda()
            else:
                return tmp[:b, :t, :e]
            
