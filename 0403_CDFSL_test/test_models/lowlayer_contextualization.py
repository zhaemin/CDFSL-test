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


class SelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.qkv = nn.Linear(input_dim, input_dim * 3)
        self.o = nn.Linear(input_dim, output_dim)
        self.init_o()
        
    def init_o(self):
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.o.bias)
        
    def forward(self, x):
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        z = F.scaled_dot_product_attention(q, k, v)
        z = self.o(z)
        
        return z

class VisionTransformer(vit.VisionTransformer):
    def __init__(self, continual_layers=None, **kwargs):
        super().__init__(**kwargs)
        
        if continual_layers != None:
            self.continual_layers = continual_layers
            print(continual_layers)
            self.st_query_pos = nn.Parameter(torch.zeros(self.patch_embed.num_patches+1, self.num_features))
            self.st_context_blocks = nn.ModuleList([SelfAttention(self.num_features, self.num_features) for i in range(len(continual_layers))])
            self.layer_nums = continual_layers
            print('continual layers:', *self.layer_nums)
    
    def forward(self, x, args):
        ways = args.train_num_ways
        x = self.prepare_tokens(x)
        sa_idx = 0
        
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            
            if self.continual_layers != None:
                if i in self.continual_layers:
                    supports = x[:ways]
                    queries = x[ways:] # 75 197 384
                    supports_cls = torch.mean(x[:ways, 0, :].view(ways, -1, self.embed_dim), dim=1) # 5 shots 384
                    prototypes = supports_cls.unsqueeze(0).repeat(queries.size(0), 1, 1) # 75 5 384
                    
                    query_pos = self.st_query_pos.unsqueeze(0).repeat(queries.size(0), 1, 1)
                    queries_with_pos = queries + query_pos
                    
                    x = torch.concat((prototypes, queries_with_pos), dim=1)
                    x = self.st_context_blocks[sa_idx](x)
                
                    queries =  queries + x[:, ways:, :]
                    x = torch.concat((supports, queries), dim=0)
                    sa_idx += 1
            
        x = self.norm(x)
        return x


class SETFSL(Baseline):
    def __init__(self, img_size, patch_size, num_objects, temperature, layer, with_cls=False, continual_layers=None, train_w_qkv=False, train_w_o=False):
        super(SETFSL, self).__init__(img_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (self.img_size // self.patch_size) ** 2
        
        self.add_cls_token = True
        self.encoder = self.load_backbone(continual_layers=continual_layers)
         
        self.encoder_dim = self.encoder.embed_dim
        self.ca_dim = self.encoder_dim
        
        self.num_objects = num_objects
        self.temperature = temperature
        self.with_cls = with_cls
        self.train_w_qkv = train_w_qkv
        self.train_w_o = train_w_o
        
        self.layer = layer
        
        #self.linear_classifier = nn.Linear(self.encoder_dim, 64)
        
        for name, param in self.encoder.named_parameters():
            if 'context_blocks' not in name and 'query_pos' not in name:
                param.requires_grad = False
            else:
                print(name)
        
        print('setfsl_lowcontext')
        print('num_object:', num_objects,' temerature:', temperature, ' layer:', layer, ' withcls:', with_cls, ' train_w_qkv:', train_w_qkv, ' train_w_o:', train_w_o,
              'ca_dim:', self.ca_dim)
        
    def load_backbone(self, patch_size=16, continual_layers=None, **kwargs):
        print('img_size: ',self.img_size)
        print('patch_size: ',self.patch_size)
        print('num_patches: ',self.num_patches)
        encoder =  VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), continual_layers=continual_layers, **kwargs)
        
        return encoder
    
    def forward(self, inputs, labels, args, device):
        '''
        x = self.encoder(inputs, args=args)[:, 0, :]
        logits = self.linear_classifier(x)
        loss = F.cross_entropy(logits, labels)
        '''
        x = self.encoder(inputs, args=args)
        tasks = split_support_query_set(x, labels, device, num_tasks=1, num_class=args.train_num_ways, num_shots=args.num_shots)
        
        loss = 0
        for x_support, x_query, y_support, y_query in tasks:
            x_support_cls = x_support[:, 0, :]
            x_query = x_query[:, 0, :]
            
            prototypes = F.normalize(torch.mean(x_support_cls.view(args.train_num_ways, args.num_shots, self.ca_dim), dim=1), dim=-1) # 5 384
            x_query = F.normalize(x_query, dim=-1) # 75 384
            
            distance = torch.einsum('qd, wd -> qw', x_query, prototypes) # 75 5
            
            logits = (distance / self.temperature).reshape(-1, args.train_num_ways)
            loss += F.cross_entropy(logits, y_query)
        
        return loss