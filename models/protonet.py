import importlib

import torch
from torch import nn
from torch.nn import functional as F

class MyModel(nn.Module):
    def __init__(self, args, network):
        super(MyModel, self).__init__()
        self.args = args
        self.encoder = network
    
    def forward(self, images, output_type='loss'):
        embeddings = self.encoder(images)
        embeddings = embeddings.view(self.args.N * (self.args.K + self.args.Q), -1)

        support_embeddings = embeddings[:self.args.N * self.args.K, :]
        query_embeddings = embeddings[self.args.N * self.args.K:, :]

        prototypes = torch.mean(support_embeddings.view(self.args.K, self.args.N, -1), dim=0)
        prototypes = F.normalize(prototypes, dim=1, p=2)

        logits = torch.mm(query_embeddings, prototypes.t()) / self.args.tau

        if output_type == 'logits':
            return logits
        elif output_type == 'loss':
            query_targets = torch.arange(self.args.N).repeat(self.args.Q).long()
            query_targets = query_targets.cuda(self.args.devices[0])
        
            loss = nn.CrossEntropyLoss()(logits, query_targets)
            return loss

    def get_network_params(self):
        modules = [self.encoder]
        for i in range(len(modules)):
            for j in modules[i].parameters():
                yield j 

    def get_other_params(self):
        modules = []
        for i in range(len(modules)):
            for j in modules[i].parameters():
                yield j