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
import backbone.vision_transformer as vit

class VisionTransformer(vit.VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class MoEFSL(nn.Module):
    def __init__(self, img_size, patch_size):
        super(MoEFSL, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.encoder = self.load_backbone()
        self.head = nn.Linear(self.encoder.num_features, 64)
        self.mlps = nn.ModuleList([
            vit.Mlp(self.encoder.num_features, hidden_features=16, out_features=self.encoder.num_features)
            for _ in range(16)
        ])
        
        self.checkencentrop = 0
        
    def load_backbone(self, patch_size=16, **kwargs):
        encoder =  VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        
        return encoder
   
    def entropy_loss(self, logits):
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        entropy = -torch.sum(probs * log_probs, dim=1)
        return entropy.mean()
    
    def forward(self, inputs, labels, args, device):
        group_idx = labels // 4
        group_idx = labels
        x = self.encoder(inputs)
        
        encoder_logit = self.head(x)
        encoder_loss = F.cross_entropy(encoder_logit, labels)
        encoder_entropy = self.entropy_loss(encoder_logit)
        self.checkencentrop = encoder_entropy
        
        out = torch.zeros_like(x)
        
        for i in range(16):
            mask = (group_idx == i)
            if mask.any():
                out[mask] = self.mlps[i](x[mask])

        logits = self.head(out)
        mlp_entropy = self.entropy_loss(logits)
        loss = F.cross_entropy(logits, labels)
        loss = loss + 0.5 * (mlp_entropy - encoder_entropy) + 0.5 * encoder_loss
        
        return loss
    
    
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