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
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__(img_size, patch_size, in_chans, num_classes, embed_dim, depth,
                         num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer)

        self.st_clsmod = nn.Parameter(torch.zeros_like(self.cls_token))

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
    
    def forward(self, x):
        x = self.prepare_tokens(x)
        
        for blk in self.blocks:
            x = blk(x)
        #without norm
        return x[:, 0]


class CLSMOD(Baseline):
    def __init__(self, img_size, patch_size):
        super(CLSMOD, self).__init__(img_size, patch_size)
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
            #logits = -torch.cdist(x_query, prototypes)
            
            prototypes = F.normalize(prototypes, dim=-1)
            x_query = F.normalize(x_query, dim=-1)
            
            distance = torch.einsum('qd, wd -> qw', x_query, prototypes) # 75 5
                
            logits = (distance / args.temperature).reshape(-1, args.test_num_ways)
            loss += F.cross_entropy(logits, y_query)
            
        return loss