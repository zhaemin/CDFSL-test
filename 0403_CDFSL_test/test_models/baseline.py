import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim
import copy
import math
import random
import numpy as np

from utils import split_support_query_set
import backbone.vision_transformer as vit

class Baseline(nn.Module):
    def __init__(self, img_size, patch_size):
        super(Baseline, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.encoder = self.load_backbone()
        print('baseline')
        
    def load_backbone(self):
        encoder = vit.__dict__['vit_small'](img_size=[self.img_size], patch_size=self.patch_size)
        
        return encoder
        
    def forward(self, inputs, labels, args, device):
        return None
    
    def fewshot_acc(self, args, inputs, labels, device):
        with torch.no_grad():
            correct = 0
            total = 0
            loss = 0
            
            x = self.encoder(inputs)
            tasks = split_support_query_set(x, labels, device, num_tasks=1, num_class=args.train_num_ways, num_shots=args.num_shots)
            
            loss = 0
            for x_support, x_query, y_support, y_query in tasks:
                prototypes = torch.mean(x_support.view(args.train_num_ways, args.num_shots, -1), dim=1)
                #logits = -torch.cdist(x_query, prototypes)
                
                prototypes = F.normalize(prototypes, dim=-1)
                x_query = F.normalize(x_query, dim=-1)
                
                distance = torch.einsum('qd, wd -> qw', x_query, prototypes) # 75 5
                    
                logits = (distance / args.temperature).reshape(-1, args.test_num_ways)
                loss += F.cross_entropy(logits, y_query)
                
                _, predicted = torch.max(logits.data, 1)
                correct += (predicted == y_query).sum().item()
                total += y_query.size(0)
                
            acc = 100 * correct / total
        return acc