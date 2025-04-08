import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim
import copy
import math
import random
import numpy as np
from functools import partial

from utils import split_support_query_set
from test_models.cls_modification import CLSMOD

class CLSMODSSL(CLSMOD):
    def __init__(self, img_size, patch_size, finetune_norm, permute_pos):
        super(CLSMODSSL, self).__init__(img_size, patch_size, finetune_norm, permute_pos)
        self.img_size = img_size
        self.patch_size = patch_size
        self.encoder = self.load_backbone()
        self.projector = self.make_mlp(self.encoder.num_features)
                
    def make_mlp(self, input_dim, hidden_dim=2048, num_layers=2, out_dim=128, last_bn=False):
        mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim)
            )
        
        for i in range(num_layers - 2):
            mlp.append(nn.Sequential(
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ))
        
        if num_layers >= 2:
            mlp.append(nn.Sequential(
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim)
            ))
        
        if last_bn:
            mlp.append(nn.BatchNorm1d(out_dim))
        
        return mlp
    
    def forward(self, inputs, labels, args, device):
        inputs[0], inputs[1] = inputs[0].to(device), inputs[1].to(device)
        labels  = labels.to(device)
        
        batch_size = inputs[0].size(0)
        print(inputs[0].shape)
        z1 = self.projector(self.encoder(inputs[0]))
        z2 = self.projector(self.encoder(inputs[1]))
        
        z1 = F.normalize(z1, p=2)
        z2 = F.normalize(z2, p=2)
        
        z = torch.cat((z1,z2), dim=0)
        logits = torch.einsum('ad, bd -> ab', z, z)
        logits.fill_diagonal_(float('-inf'))
        
        tmp_labels1 = torch.arange(batch_size, 2*batch_size)
        tmp_labels2 = torch.arange(0, batch_size)
        labels = torch.cat((tmp_labels1, tmp_labels2)).to(device)
        
        loss = F.cross_entropy(logits / 0.5, labels)
        
        return loss