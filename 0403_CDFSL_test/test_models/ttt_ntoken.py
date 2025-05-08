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
from test_models.baseline import Baseline

class VisionTransformer(vit.VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def prepare_tokens(self, x, ntoken=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding
        
        cls_tokens = self.cls_token + ntoken
        # add the [CLS] token to the embed patch tokens
        cls_tokens = cls_tokens.expand(B, -1, -1)
        
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)
        
        return self.pos_drop(x)
    
    def forward(self, x, ntoken=None):
        x = self.prepare_tokens(x, ntoken)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        return x[:, 0]

class TTTSSL(nn.Module):
    def __init__(self, img_size, patch_size):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.encoder = self.load_backbone()
        self.ntoken = nn.Parameter(torch.zeros(1, 1, self.encoder.num_features))
        self.fix_encoder()
        
    def load_backbone(self, patch_size=16, **kwargs):
        encoder =  VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        
        return encoder

    def fix_encoder(self):
        for name, param in self.encoder.named_parameters():
            if 'norm' in name or 'st_' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def calculate_logits(self, args, x_support, x_query, ntoken):
        x_support = self.encoder(x_support, ntoken)
        prototypes = torch.mean(x_support.view(args.train_num_ways, args.num_shots, -1), dim=1)
        prototypes = F.normalize(prototypes, dim=-1)
        
        x_query = self.encoder(x_query, ntoken)
        x_query = F.normalize(x_query, dim=-1)
        
        logits = torch.einsum('qd, wd -> qw', x_query, prototypes) # 75 5
        
        return logits
    
    def forward(self, inputs, labels, args, device):
        tasks = split_support_query_set(inputs, labels, device, num_tasks=1, num_class=args.train_num_ways, num_shots=args.num_shots)
        
        total_loss = 0
        for x_support, x_query, y_support, y_query in tasks:
            logits1 = self.calculate_logits(args, x_support[:, 0], x_query[:, 0], self.ntoken)
            logits2 = self.calculate_logits(args, x_support[:, 1], x_query[:, 1], self.ntoken)
            
            loss = (self.cross_entropyloss(logits1, logits2) + self.cross_entropyloss(logits2, logits1)) / 2
            loss += (F.cross_entropy(logits1, y_query) + F.cross_entropy(logits2, y_query)) / 2
            total_loss += loss
        
        return total_loss
    
    def cross_entropyloss(self, logits1, logits2):
        logits2 = logits2.detach()
        p1_log = F.log_softmax(logits1 / 0.1, dim=1)
        p2 = F.softmax(logits2 / 0.04, dim=1)
        
        loss = - (p2 * p1_log).sum(dim=1).mean()
        return loss
    
    def fewshot_acc(self, args, inputs, labels, device):
        tasks = split_support_query_set(inputs, labels, device, num_tasks=1, num_class=args.train_num_ways, num_shots=args.num_shots)
        
        query_size = args.num_shots * args.num_queries
        n_prompt = 1
        #ntoken = nn.Parameter(torch.zeros(1, n_prompt, self.encoder.num_features, device=device))
        ntoken = copy.deepcopy(self.ntoken)
                
        optimizer = optim.AdamW([
            {'params': [ntoken], 'lr': 0.01},
        ])
        
        correct = 0
        loss = 0
        total = 0
        n_iters = 10
        
        for x_support, x_query, y_support, y_query in tasks:
            self.train()
            
            for _ in range(n_iters):
                logits1 = self.calculate_logits(args, x_support[:, 0], x_query[:, 0], ntoken) / 0.1
                logits2 = self.calculate_logits(args, x_support[:, 1], x_query[:, 1], ntoken) / 0.04
                
                loss = (self.cross_entropyloss(logits1, logits2) + self.cross_entropyloss(logits2, logits1)) / 2
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            self.eval()
            
            
            with torch.no_grad():
                x_support, x_query = x_support[:, 2], x_query[:, 2]
                distance = self.calculate_logits(args, x_support, x_query, ntoken)
                logits = (distance / args.temperature).reshape(-1, args.test_num_ways)
                _, predicted = torch.max(logits.data, 1)
                correct += (predicted == y_query).sum().item()
                total += y_query.size(0)
        
            acc = 100 * correct / total
            #print(acc)
        return acc