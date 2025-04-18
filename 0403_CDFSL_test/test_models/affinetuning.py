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
from backbone.vit_utils import trunc_normal_
from test_models.baseline import Baseline

class VisionTransformer(vit.VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        num_heads = kwargs['num_heads']
        num_layers = kwargs['depth']
        self.only_lastnorm = kwargs['only_lastnorm']
        self.temperature = 1
        
        if self.only_lastnorm:
            self.st_affine_weight_projector = nn.ModuleList([
                nn.Linear(self.num_features, self.num_features)
                for _ in range((1))
            ])
            self.st_affine_bias_projector = nn.ModuleList([
                nn.Linear(self.num_features, self.num_features)
                for _ in range((1))
            ])
        else:
            self.st_affine_weight_projector = nn.ModuleList([
                nn.Linear(self.num_features, self.num_features)
                for _ in range((num_layers*2+1))
            ])
            self.st_affine_bias_projector = nn.ModuleList([
                nn.Linear(self.num_features, self.num_features)
                for _ in range((num_layers*2+1))
            ])
            
        self.st_clsmod = nn.Parameter(torch.zeros_like(self.cls_token))
        
        for proj in self.st_affine_weight_projector:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

        for proj in self.st_affine_bias_projector:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)
    
    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        cls_tokens = self.cls_token + self.st_clsmod
        # add the [CLS] token to the embed patch tokens
        cls_tokens = cls_tokens.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        pos_embed =  self.interpolate_pos_encoding(x, w, h)
        x = x + pos_embed
        
        return self.pos_drop(x)
    
    def forward(self, x, prototypes=None):
        x = self.prepare_tokens(x)
        
        if prototypes != None:
            prototypes = prototypes.sum(dim=0) # 1 dim
            for i, blk in enumerate(self.blocks):
                if self.only_lastnorm:
                    x = blk(x)
                else:
                    weight1 = self.st_affine_weight_projector[2*i](prototypes) * self.temperature # dim
                    bias1 = self.st_affine_bias_projector[2*i](prototypes) * self.temperature 
                    weight2 = self.st_affine_weight_projector[2*i+1](prototypes) * self.temperature 
                    bias2 = self.st_affine_bias_projector[2*i+1](prototypes) * self.temperature 
                    
                    x = blk.forward_affinetuning(x, weight1, bias1, weight2, bias2)
                
            weight = self.st_affine_weight_projector[-1](prototypes) * self.temperature 
            bias = self.st_affine_weight_projector[-1](prototypes) * self.temperature 
            
            x = vit.dynamic_affine_norm(x, self.num_features, self.norm.weight+weight, self.norm.bias+bias)
        else:
            for i, blk in enumerate(self.blocks):
                x = blk(x)
            x = self.norm(x)
             
        return x[:, 0]

class AffineTuning(Baseline):
    def __init__(self, img_size, patch_size, only_lastnorm):
        super(AffineTuning, self).__init__(img_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.encoder = self.load_backbone(patch_size, only_lastnorm)

        self.fix_encoder()

    def fix_encoder(self):
        for name, param in self.encoder.named_parameters():
            if 'st_' not in name:
                param.requires_grad = False
            else:
                print(name)
    
    def load_backbone(self, patch_size=16, only_lastnorm=False, **kwargs):
        encoder =  VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), only_lastnorm=only_lastnorm, **kwargs) # affine false로 설정
        
        return encoder
    
    def calculate_distance(self, args, x_support, x_query):
        x_support = self.encoder(x_support)
        prototypes = torch.mean(x_support.view(args.train_num_ways, args.num_shots, -1), dim=1)
        prototypes = F.normalize(prototypes, dim=-1)
        
        x_query = self.encoder(x_query, prototypes)
        x_query = F.normalize(x_query, dim=-1)
        
        distance = torch.einsum('qd, wd -> qw', x_query, prototypes) # 75 5
        
        return distance
    
    def forward(self, inputs, labels, args, device):
        tasks = split_support_query_set(inputs, labels, device, num_tasks=1, num_class=args.train_num_ways, num_shots=args.num_shots)

        loss = 0
        for x_support, x_query, y_support, y_query in tasks:
            distance = self.calculate_distance(args, x_support, x_query)
            logits = (distance / args.temperature).reshape(-1, args.test_num_ways)
            loss += F.cross_entropy(logits, y_query)
            
        return loss
    
    def fewshot_acc(self, args, inputs, labels, device):
        with torch.no_grad():
            correct = 0
            total = 0
            loss = 0
            
            tasks = split_support_query_set(inputs, labels, device, num_tasks=1, num_class=args.train_num_ways, num_shots=args.num_shots)
            
            loss = 0
            for x_support, x_query, y_support, y_query in tasks:
                distance = self.calculate_distance(args, x_support, x_query)
                logits = (distance / args.temperature).reshape(-1, args.test_num_ways)
                loss += F.cross_entropy(logits, y_query)
                
                _, predicted = torch.max(logits.data, 1)
                correct += (predicted == y_query).sum().item()
                total += y_query.size(0)
                
            acc = 100 * correct / total
        return acc