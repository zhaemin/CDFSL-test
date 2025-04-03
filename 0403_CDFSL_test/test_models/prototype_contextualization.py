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


class CrossAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.q = nn.Linear(input_dim, output_dim)
        self.k = nn.Linear(input_dim, output_dim)
        self.v = nn.Linear(input_dim, output_dim)
        self.o = nn.Linear(output_dim, output_dim)
        self.init_o()
        
    def init_o(self):
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.o.bias)
    
    def forward(self, x, y):
        q = self.q(x)
        k = self.k(y)
        v = self.v(y)
        
        z = F.scaled_dot_product_attention(q, k, v)
        z = self.o(z)
        
        return z

class SelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.qkv = nn.Linear(input_dim, input_dim * 3)
        self.o = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        z = F.scaled_dot_product_attention(q, k, v)
        z = self.o(z)
        
        return z

class VisionTransformer(vit.VisionTransformer):
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, continual_layers=None, **kwargs):
        super().__init__(img_size, patch_size, in_chans, num_classes, embed_dim, depth,
                         num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer)

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        #x = self.norm(x)
        # modified
        return x

class SETFSL(nn.Module):
    def __init__(self, img_size, patch_size, num_objects, temperature, layer, with_cls=False, continual_layers=None, train_w_qkv=False, train_w_o=False):
        super(SETFSL, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (self.img_size // self.patch_size) ** 2
        
        self.add_cls_token = True
        self.encoder = self.load_backbone()
         
        self.encoder_dim = self.encoder.embed_dim
        self.ca_dim = self.encoder_dim
        
        self.num_objects = num_objects
        self.temperature = temperature
        self.with_cls = with_cls
        self.continual_layers = continual_layers
        self.train_w_qkv = train_w_qkv
        self.train_w_o = train_w_o
        
        self.layer = layer
        
        self.norm = nn.LayerNorm(self.encoder_dim)
        
        # individual CA
        self.crossattn = CrossAttention(self.encoder_dim, self.ca_dim)
        
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False
        
        print('prototype_contextualization')
        print('num_object:', num_objects,' temerature:', temperature, ' layer:', layer, ' withcls:', with_cls, ' train_w_qkv:', train_w_qkv, ' train_w_o:', train_w_o,
              'ca_dim:', self.ca_dim)
        
    def load_backbone(self, patch_size=16, **kwargs):
        encoder =  VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        
        return encoder
        
    def forward(self, inputs, labels, args, device):
        tasks = split_support_query_set(inputs, labels, device, num_tasks=1, num_class=args.train_num_ways, num_shots=args.num_shots)
        
        loss = 0
        for x_support, x_query, y_support, y_query in tasks:
            x_support = self.encoder(x_support)
            x_query = self.encoder(x_query)
            query_size = x_query.size(0)
            
            x_support_cls = x_support[:, 0, :].unsqueeze(1)
            prototypes_cls = F.normalize(torch.mean(x_support_cls.view(args.train_num_ways, args.num_shots, self.ca_dim), dim=1), dim=-1) # 5 384
            
            query_cls = x_query[:, 0, :].unsqueeze(1)# queries 1 384
            prototypes_cls = prototypes_cls.unsqueeze(0).repeat(query_size, 1, 1) # queries 5 384
            q = torch.concat((prototypes_cls, query_cls), dim=1)
            
            # context block
            contextualized_x  = q + self.crossattn(q, x_query)
            
            prototypes = self.norm(contextualized_x[:, :-1, :]) # 75 5 384
            x_query = self.norm(contextualized_x[:, -1, :]).unsqueeze(1)  # 75 1 384
            
            prototypes = F.normalize(prototypes, dim=-1) # 75 5 384
            x_query = F.normalize(x_query, dim=-1) # 75 1 384
            
            distance = torch.einsum('bqd, bwd -> bqw', x_query, prototypes) # 75 5
            
            logits = (distance / self.temperature).reshape(-1, args.train_num_ways)
            loss += F.cross_entropy(logits, y_query)
            
        return loss
    
    def fewshot_acc(self, args, inputs, labels, device):
        with torch.no_grad():
            correct = 0
            total = 0
            loss = 0
            
            tasks = split_support_query_set(inputs, labels, device, num_tasks=1, num_class=args.test_num_ways, num_shots=args.num_shots)
            
            for x_support, x_query, y_support, y_query in tasks:
                x_support = self.encoder(x_support)
                x_query = self.encoder(x_query)
                query_size = x_query.size(0)
                
                x_support_cls = x_support[:, 0, :].unsqueeze(1)
                prototypes_cls = F.normalize(torch.mean(x_support_cls.view(args.train_num_ways, args.num_shots, self.ca_dim), dim=1), dim=-1) # 5 384
                
                query_cls = x_query[:, 0, :].unsqueeze(1)# queries 1 384
                prototypes_cls = prototypes_cls.unsqueeze(0).repeat(query_size, 1, 1) # queries 5 384
                q = torch.concat((prototypes_cls, query_cls), dim=1)
                
                # context block
                contextualized_x  = q + self.crossattn(q, x_query)
                
                prototypes = self.norm(contextualized_x[:, :-1, :]) # 75 5 384
                x_query = self.norm(contextualized_x[:, -1, :]).unsqueeze(1)  # 75 1 384
                
                prototypes = F.normalize(prototypes, dim=-1) # 75 5 384
                x_query = F.normalize(x_query, dim=-1) # 75 1 384
                
                distance = torch.einsum('bqd, bwd -> bqw', x_query, prototypes) # 75 5
                
                logits = (distance / self.temperature).reshape(-1, args.test_num_ways)
                loss += F.cross_entropy(logits, y_query)
                
                _, predicted = torch.max(logits.data, 1)
                correct += (predicted == y_query).sum().item()
                total += y_query.size(0)
                
            acc = 100 * correct / total
        return acc