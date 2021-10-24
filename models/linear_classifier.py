import torch
from torch import nn
from torch.nn import functional as F

class MyModel(nn.Module):
    def __init__(self, args, network, out_dimension):
        super(MyModel, self).__init__()
        self.args = args
        self.encoder = network
        self.out_dimension = out_dimension

        # determine input features
        if args.network_name == 'convnet':
            in_dimension = 64
        elif args.network_name == 'resnet':
            in_dimension = 640
        
        self.fc = nn.Linear(in_features=in_dimension, out_features=out_dimension)
    
    def forward(self, images):
        embeddings = self.encoder(images)
        logits = self.fc(embeddings)
        return logits
    
    def get_network_params(self):
        modules = [self.encoder]
        for i in range(len(modules)):
            for j in modules[i].parameters():
                yield j 

    def get_other_params(self):
        modules = [self.fc]
        for i in range(len(modules)):
            for j in modules[i].parameters():
                yield j