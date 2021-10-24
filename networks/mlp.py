from collections import OrderedDict

from torch import nn
from torch.nn import functional as F

class MyNetwork(nn.Module):
    def __init__(self, args, input_dim=2, output_dim=100, n_hidden=100):
        super(MyNetwork, self).__init__()
        self.args = args
        self.layer1 = nn.Linear(input_dim, n_hidden)
        self.layer2 = nn.Linear(n_hidden, n_hidden)
        self.layer3 = nn.Linear(n_hidden, n_hidden)
        self.layer4 = nn.Linear(n_hidden, output_dim)

    def forward(self, x, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        x = F.linear(x, weight=params['layer1.weight'], bias=params['layer1.bias'])
        x = F.relu(x)
        x = F.linear(x, weight=params['layer2.weight'], bias=params['layer2.bias'])
        x = F.relu(x)
        x = F.linear(x, weight=params['layer3.weight'], bias=params['layer3.bias'])
        x = F.relu(x)
        x = F.linear(x, weight=params['layer4.weight'], bias=params['layer4.bias'])    
        return x