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

    def get_intermediate_layers(self, x, n=11):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == n:
                x = self.norm(x)
                return x[:, 0]
        return x

class TTTSSL(nn.Module):
    def __init__(self, img_size, patch_size):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.encoder = self.load_backbone()
        self.projector = self.make_mlp(self.encoder.num_features)
        
    def load_backbone(self, patch_size=16, **kwargs):
        encoder =  VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        
        return encoder
                
    def make_mlp(self, input_dim, hidden_dim=2048, num_layers=2, out_dim=128, last_bn=False):
        mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim)
            )
        
        for i in range(num_layers - 2):
            mlp.append(nn.Sequential(
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ))
        
        if num_layers >= 2:
            mlp.append(nn.Sequential(
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim)
            ))
        
        if last_bn:
            mlp.append(nn.BatchNorm1d(out_dim))
        
        return mlp
    
    
    def ssl_forward(self, inputs, device):
        inputs[0], inputs[1] = inputs[0].to(device), inputs[1].to(device)
        
        batch_size = inputs[0].size(0)
        z1 = self.projector(self.encoder.get_intermediate_layers(inputs[0], n=11))
        z2 = self.projector(self.encoder.get_intermediate_layers(inputs[1], n=11))
        
        z1 = F.normalize(z1, p=2)
        z2 = F.normalize(z2, p=2)
        
        z = torch.cat((z1,z2), dim=0)
        logits = torch.einsum('ad, bd -> ab', z, z)
        logits.fill_diagonal_(float('-inf'))
        
        tmp_labels1 = torch.arange(batch_size, 2*batch_size)
        tmp_labels2 = torch.arange(0, batch_size)
        labels = torch.cat((tmp_labels1, tmp_labels2)).to(device)
        
        loss = F.cross_entropy(logits / 0.5, labels)
        
        return loss

    def calculate_distance(self, args, x_support, x_query):
        x_support = self.encoder(x_support)
        prototypes = torch.mean(x_support.view(args.train_num_ways, args.num_shots, -1), dim=1)
        prototypes = F.normalize(prototypes, dim=-1)
        
        x_query = self.encoder(x_query)
        x_query = F.normalize(x_query, dim=-1)
        
        distance = torch.einsum('qd, wd -> qw', x_query, prototypes) # 75 5
        
        return distance
    
    def fewshot_acc(self, args, inputs, labels, device):
        tasks = split_support_query_set(inputs, labels, device, num_tasks=1, num_class=args.train_num_ways, num_shots=args.num_shots)
        
        net = copy.deepcopy(self)
        
        '''
        norm_param = []
        cls_param = []
        for name, param in net.encoder.named_parameters():
            if 'norm' in name:
                norm_param.append(param)
                continue
            if 'st_' in name:
                cls_param.append(param)
                continue
            param.requires_grad = False
        
        
        optimizer = optim.AdamW([
            {'params': norm_param, 'lr': 1e-6},
            {'params': cls_param, 'lr': 0.01}
        ])
        '''

        optimizer = optim.AdamW([
            {'params': net.parameters(), 'lr': 1e-6},
        ])
        
        correct = 0
        loss = 0
        total = 0
        n_iters = 50
        
        for x_support, x_query, y_support, y_query in tasks:
            net.train()
            for _ in range(n_iters):
                loss = self.ssl_forward(x_support, device)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            net.eval()
            
            with torch.no_grad():
                x_support, x_query = x_support[:, 2], x_query[:, 2]
                distance = net.calculate_distance(args, x_support, x_query)
                logits = (distance / args.temperature).reshape(-1, args.test_num_ways)
                _, predicted = torch.max(logits.data, 1)
                correct += (predicted == y_query).sum().item()
                total += y_query.size(0)
        
            acc = 100 * correct / total
            #print(acc)
        return acc
