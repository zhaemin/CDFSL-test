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
    def __init__(self, permute_pos, **kwargs):
        super().__init__(**kwargs)
        self.permute_pos = permute_pos
        self.st_clsmod = nn.Parameter(torch.zeros_like(self.cls_token))
        
    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        cls_tokens = self.cls_token + self.st_clsmod
        # add the [CLS] token to the embed patch tokens
        cls_tokens = cls_tokens.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        pos_embed =  self.interpolate_pos_encoding(x, w, h)
        
        if self.permute_pos:
            class_pos_embed = pos_embed[:, 0]
            patch_pos_embed = pos_embed[:, 1:]
            perm = torch.randperm(patch_pos_embed.shape[1])
            patch_pos_embed = patch_pos_embed[:, perm, :]
            pos_embed =  torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
        
        x = x + pos_embed
        
        return self.pos_drop(x)
    
    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        return x[:, 0]


class CLSMOD(Baseline):
    def __init__(self, img_size, patch_size, finetune_norm, permute_pos):
        super(CLSMOD, self).__init__(img_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.encoder = self.load_backbone(patch_size, permute_pos)
        self.fix_encoder(finetune_norm)

    def fix_encoder(self, finetune_norm):
        if finetune_norm:
            for name, param in self.encoder.named_parameters():
                if 'norm' in name or 'st_' in name:
                    print(name)
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            for name, param in self.encoder.named_parameters():
                if 'st_' not in name:
                    param.requires_grad = False
                else:
                    print(name)
    
    def load_backbone(self, patch_size=16, permute_pos=False, **kwargs):
        encoder =  VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), permute_pos=permute_pos, **kwargs)
        
        return encoder
        
    def forward(self, inputs, labels, args, device):
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
            
        return loss