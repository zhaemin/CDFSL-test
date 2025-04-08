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

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=8):
        super().__init__()
        assert output_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.output_dim = output_dim

        self.q = nn.Linear(input_dim, output_dim)
        self.k = nn.Linear(input_dim, output_dim)
        self.v = nn.Linear(input_dim, output_dim)
        self.o = nn.Linear(output_dim, output_dim)
        
        self.init_o()

    def init_o(self):
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.o.bias)

    def forward(self, x, y):
        B, N, _ = x.shape
        _, M, _ = y.shape

        q = self.q(x)
        k = self.k(y)
        v = self.v(y)

        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)
        k = k.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        z = F.scaled_dot_product_attention(q, k, v)

        # Concatenate heads
        z = z.transpose(1, 2).contiguous().view(B, N, self.output_dim)  # (B, N, output_dim)
        
        out = self.o(z)
        return out


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

class ContextUnit(nn.Module):
    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.norm1 = nn.LayerNorm(input_dim)
        self.crossattn = nn.ModuleList([
            MultiHeadCrossAttention(input_dim, input_dim)
            for _ in range(num_layers)
        ])
        #self.norm2 = nn.LayerNorm(input_dim)
        #self.mlp = vit.Mlp(in_features=input_dim, hidden_features=input_dim*4, out_features=input_dim)
    
    def forward(self, q, kv):
        '''
        r1 = q
        q1 = self.norm1(q)
        for layer in self.crossattn:
            q1 = layer(q1, kv)
        q1 = r1 + q1
        '''
        q1 = q
        for layer in self.crossattn:
            q1 = q1 + layer(q1, kv)
        
        return  q1
        

class VisionTransformer(vit.VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        #x = self.norm(x)
        return x
    
    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(x)
        return output

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
        
        self.context_unit = ContextUnit(self.encoder_dim, self.layer)
        print(len(self.context_unit.crossattn))
        self.fix_encoder()
        
        print('prototype_contextualization')
        print('num_object:', num_objects,' temerature:', temperature, ' layer:', layer, ' withcls:', with_cls, ' train_w_qkv:', train_w_qkv, ' train_w_o:', train_w_o,
              'ca_dim:', self.ca_dim)
        
    def fix_encoder(self):
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False
        
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
            
            contextualized_x = self.context_unit(q, x_query)
            
            prototypes = contextualized_x[:, :-1, :] # 75 5 384
            x_query = contextualized_x[:, -1, :].unsqueeze(1)  # 75 1 384
            
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
                
                contextualized_x = self.context_unit(q, x_query)
                
                prototypes = contextualized_x[:, :-1, :] # 75 5 384
                x_query = contextualized_x[:, -1, :].unsqueeze(1)  # 75 1 384
                
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