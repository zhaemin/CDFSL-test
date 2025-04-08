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

        self.st_posmod = nn.Parameter(torch.zeros_like(self.pos_embed))
    
    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            # modified
            pos_embed = self.pos_embed + self.st_posmod
            return pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        
        #modified
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1) + self.posmod
    
    def forward(self, x):
        pos_embed = self.st_posmod.data + self.st_posmod
        
        x = self.prepare_tokens(x)
        
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]


class POSMOD(Baseline):
    def __init__(self, img_size, patch_size):
        super(POSMOD, self).__init__(img_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.encoder = self.load_backbone()
        
        for name, param in self.encoder.named_parameters():
            if 'st_' not in name:
                param.requires_grad = False
            else:
                print(name)
        
    def load_backbone(self, patch_size=16, **kwargs):
        encoder =  VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        
        return encoder
        
    def forward(self, inputs, labels, args, device):
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
            
        return loss