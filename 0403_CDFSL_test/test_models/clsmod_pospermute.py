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
from test_models.cls_modification import CLSMOD

class VisionTransformer(vit.VisionTransformer):
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__(img_size, patch_size, in_chans, num_classes, embed_dim, depth,
                         num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer)

        self.st_clsmod = nn.Parameter(torch.zeros_like(self.cls_token))

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            # modified
            class_pos_embed = self.pos_embed[:, 0]
            patch_pos_embed = self.pos_embed[:, 1:]
            
            perm = torch.randperm(patch_pos_embed.shape[1])
            patch_pos_embed = patch_pos_embed[:, perm, :]
            
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
        
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
        perm = torch.randperm(patch_pos_embed.shape[1])
        patch_pos_embed = patch_pos_embed[:, perm, :]
        
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
    
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


class CLSMOD(CLSMOD):
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