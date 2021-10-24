import importlib

import torch
from torch import nn
from torch.nn import functional as F

class MyModel(nn.Module):
    def __init__(self, args, network, target_model, target_task):
        super(MyModel, self).__init__()
        self.args = args
        self.encoder = network
        self.target_model = target_model
        self.target_task = target_task

        self.label_map = torch.zeros(1 + max(target_task)) - 1
        self.label_map = self.label_map.long().cuda(args.devices[0])
        for i, x in enumerate(target_task):
            self.label_map[x] = i
    
    def forward(self, images, flag_in_task=False, output_type='loss', labels=-1):
        embeddings = self.encoder(images) 
        embeddings = embeddings.view(self.args.N * (self.args.K + self.args.Q), -1)

        support_embeddings = embeddings[:self.args.N * self.args.K, :]
        query_embeddings = embeddings[self.args.N * self.args.K:, :]

        prototypes = torch.mean(support_embeddings.view(self.args.K, self.args.N, -1), dim=0)
        prototypes = F.normalize(prototypes, dim=1, p=2)

        support_logits = torch.mm(support_embeddings, prototypes.t()) / self.args.tau
        query_logits = torch.mm(query_embeddings, prototypes.t()) / self.args.tau

        if not flag_in_task:
            if output_type == 'logits':
                return query_logits
            elif output_type == 'loss':
                query_targets = torch.arange(self.args.N).repeat(self.args.Q).long()
                query_targets = query_targets.cuda(self.args.devices[0])
                loss = nn.CrossEntropyLoss()(query_logits, query_targets)
                return loss
        else:
            if output_type == 'logits':
                return query_logits
            elif output_type == 'loss':
                with torch.no_grad():
                    target_logits = self.target_model(images)[:, self.label_map[labels[:self.args.N]]]
                    target_support_logits = target_logits[:self.args.N * self.args.K, :]
                    target_query_logits = target_logits[self.args.N * self.args.K, :]
                support_targets = torch.arange(self.args.N).repeat(self.args.K).long()
                support_targets = support_targets.cuda(self.args.devices[0])
                support_loss = nn.CrossEntropyLoss()(support_logits, support_targets)
                distill_loss = self.args.lambd * nn.KLDivLoss(reduction='batchmean')(
                    F.log_softmax(support_logits, dim=1), F.softmax(target_support_logits, dim=1)
                )
                loss = support_loss + distill_loss
                return loss, support_loss, distill_loss

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