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
        num_heads = kwargs['num_heads']
        num_layers = kwargs['depth']
        #self.st_clsmod = nn.Parameter(torch.zeros_like(self.cls_token))
        
        self.st_prefix_projector = nn.ModuleList([
            nn.Linear(self.num_features, self.num_features)
            for _ in range(num_layers)
        ])
        
        for proj in self.st_prefix_projector:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)
    '''
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
    '''
    def forward(self, x, prefix_prompt=None):
        x = self.prepare_tokens(x)
        projector_idx = 0
        for i, blk in enumerate(self.blocks):
            if  prefix_prompt == None:
                x = blk.forward_prefixtuning(x, blk_num=i)
            else:
                projected_prefix_prompt = self.st_prefix_projector[projector_idx](prefix_prompt)
                projector_idx += 1
                x = blk.forward_prefixtuning(x, prefix_prompt=projected_prefix_prompt, blk_num=i)
        x = self.norm(x)
        
        return x[:, 0]
    
    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk.forward_prefixtuning(x, blk_num=i)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x[:, 0]))
        return output

class PrefixAttention(vit.Attention):
    def forward(self, x, prefix_tokens=None, blk_num=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if prefix_tokens is not None:
            '''
            prefix_tokens = prefix_tokens.unsqueeze(0).unsqueeze(0)
            prefix_tokens = prefix_tokens.repeat(B, self.num_heads, 1, 1)
            
            prefix_k = prefix_tokens
            prefix_v = prefix_tokens
            k = torch.cat([prefix_k, k], dim=2)  # [B, num_heads, prefix_len + N, head_dim]
            v = torch.cat([prefix_v, v], dim=2)
            '''
            
            prefix_tokens = prefix_tokens.unsqueeze(0).expand(B, -1, -1)  # B, prefix_len, dim
            
            prefix_qkv = self.qkv(prefix_tokens)
            prefix_qkv = prefix_qkv.reshape(B, prefix_tokens.shape[1], 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            prefix_k, prefix_v = prefix_qkv[1], prefix_qkv[2] # kv projected tokens 

            # k, v에 prefix 붙이기
            k = torch.cat([prefix_k, k], dim=2)
            v = torch.cat([prefix_v, v], dim=2)
            

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if blk_num > 8:
            attn = attn * 0
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

    
class PrefixTuning(Baseline):
    def __init__(self, img_size, patch_size):
        super(PrefixTuning, self).__init__(img_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.encoder = self.load_backbone(patch_size)

        self.fix_encoder()

    def fix_encoder(self):
        for name, param in self.encoder.named_parameters():
            '''
            if ('blocks.8' in name or 'blocks.9' in name or 'blocks.10' in name or 'blocks.11' in name or 'st_' in name or ('norm' in name and 'blocks' not in name)):
                param.requires_grad = True
                print(name)
            '''
            if 'st_' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    def calculate_distance(self, args, x_support, x_query):
        x_support = self.encoder.get_intermediate_layers(x_support, n=4) # layer 8~11
        prototypes = torch.mean(x_support[-1].view(args.train_num_ways, args.num_shots, -1), dim=1)
        prototypes = F.normalize(prototypes, dim=-1)
        
        x_query = self.encoder(x_query, prototypes)
        x_query = F.normalize(x_query, dim=-1)
        prototypes = torch.mean(x_support[-1].view(args.train_num_ways, args.num_shots, -1), dim=1)
        prototypes = F.normalize(prototypes, dim=-1)
        
        distance = torch.einsum('qd, wd -> qw', x_query, prototypes) # 75 5
        
        return distance
    
    
    def load_backbone(self, patch_size=16, **kwargs):
        encoder =  VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), attn_class=PrefixAttention, **kwargs)
        
        return encoder
        
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