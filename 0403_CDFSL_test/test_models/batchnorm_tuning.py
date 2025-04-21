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


class CustomBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mean=None, var=None):
        if mean is None:
            mean = torch.mean(x, dim=0)
        if var is None:
            var = torch.var(x, dim=0, unbiased=False)
        x_norm = (x - mean) / ((var + self.eps) ** 0.5)
        out = self.weight * x_norm + self.bias
        return out, mean, var

class BNBlock(vit.Block):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_class=vit.Attention):
        super().__init__(dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_class=vit.Attention)
        self.bn1 = CustomBatchNorm(dim)
        self.bn2 = CustomBatchNorm(dim)
        
    def forward(self, x, return_attention=False, prototype_stat=None):
        mean1, var1, mean2, var2 = None, None, None, None
        if prototype_stat:
            mean1, var1, mean2, var2 = prototype_stat[0], prototype_stat[1], prototype_stat[2], prototype_stat[3]
        
        x_bn, mean1, var1 = self.bn1(x, mean1, var1)
        x = x_bn + self.norm1(x)
        y, attn = self.attn(x)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x_bn, mean2, var2  = self.bn2(x, mean2, var2)
        x = x_bn + self.norm2(x)
        x = x + self.drop_path(self.mlp(x))
        
        return x, mean1, var1, mean2, var2


class VisionTransformer(vit.VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.st_clsmod = nn.Parameter(torch.zeros_like(self.cls_token))
        self.prototype_stat = [None for _ in range(len(self.blocks))]
    
    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        cls_tokens = self.cls_token + self.st_clsmod
        # add the [CLS] token to the embed patch tokens
        cls_tokens = cls_tokens.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)
        
        return self.pos_drop(x)
    
    def forward(self, x, prototype_stat=False):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if prototype_stat: # queries forward
                assert self.prototype_stat[i] != None, 'prototype stat is empty'
                x, mean1, var1, mean2, var2 = blk(x, prototype_stat = self.prototype_stat[i])
            else: # supports forward
                x, mean1, var1, mean2, var2 = blk(x)
                self.prototype_stat[i] = [mean1, var1, mean2, var2]
                
        x = self.norm(x)
        
        return x[:, 0]


class BNTuning(Baseline):
    def __init__(self, img_size, patch_size):
        super(BNTuning, self).__init__(img_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.encoder = self.load_backbone(patch_size)
        self.fix_encoder()

    def fix_encoder(self):
        for name, param in self.encoder.named_parameters():
            if 'norm' in name or 'st_' in name or 'bn' in name:
                print(name)
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    def load_backbone(self, patch_size=16, **kwargs):
        encoder =  VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), block_class=BNBlock, **kwargs)
        
        return encoder

    def calculate_distance(self, args, x_support, x_query):
        x_support = self.encoder(x_support)
        prototypes = torch.mean(x_support.view(args.train_num_ways, args.num_shots, -1), dim=1)
        prototypes = F.normalize(prototypes, dim=-1)
        
        x_query = self.encoder(x_query, prototype_stat=True)
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