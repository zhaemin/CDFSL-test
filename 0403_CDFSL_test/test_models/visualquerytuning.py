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

from tqdm import tqdm

from utils import split_support_query_set
import backbone.vision_transformer as vit
from test_models.baseline import Baseline

class VisionTransformer(vit.VisionTransformer):
    def __init__(self, num_query_tokens, **kwargs):
        super().__init__(**kwargs)
        self.num_query_tokens = num_query_tokens
        print('nqt:',self.num_query_tokens)
        self.st_query_tokens = nn.Parameter(torch.zeros(len(self.blocks), 1, num_query_tokens, self.num_features))
        self.depth = len(self.blocks)
        
        '''
        self.st_prototype_projector = nn.ModuleList([
            nn.Linear(self.num_features, self.num_features)
            for _ in range(self.depth)
        ])
        '''
            
    def forward(self, x, prototype=None):
        query_outputs = []
        B = x.size(0)
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if prototype == None:
                x = torch.cat((self.st_query_tokens[i].repeat(B, 1, 1), x), dim=1)
                x = blk.forward_vqt(x, num_query_tokens=self.num_query_tokens)
                query_outputs.append(x[:, :self.num_query_tokens])
                x = x[:, self.num_query_tokens:]
            else:
                query_token = torch.cat((self.st_query_tokens[i], self.st_prototype_projector[i](prototype)), dim=1)
                x = torch.cat((query_token.repeat(B, 1, 1), x), dim=1)
                x = blk.forward_vqt(x, num_query_tokens=self.num_query_tokens + 5)
                query_outputs.append(x[:, :self.num_query_tokens+5])
                x = x[:, self.num_query_tokens+5:]
            
        x = self.norm(x)
        
        return x[:, 0], query_outputs


class VQTAttention(vit.Attention):
    def forward(self, x, num_query_tokens):
        B, N, C = x.shape
        
        x_kv = x[:, num_query_tokens:]
        qkv = self.qkv(x_kv).reshape(B, N-num_query_tokens, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = qkv[1], qkv[2]
        qkv_for_q = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = qkv_for_q[0]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale # B h ql+N dim 
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class VQTuning(Baseline):
    def __init__(self, img_size, patch_size, num_query_tokens):
        super(VQTuning, self).__init__(img_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.encoder = self.load_backbone(patch_size, num_query_tokens=num_query_tokens)
        self.layer_weight = nn.Parameter(torch.ones(self.encoder.depth))
        self.head = nn.Linear(self.encoder.num_features * (self.encoder.depth+1), 5)
        
        self.fix_encoder()

    def fix_encoder(self):
        for name, param in self.encoder.named_parameters():
            if 'st_' in name:
                param.requires_grad = True
                print(name)
            else:
                param.requires_grad = False
    
    def calculate_distance(self, args, x_support, x_query):
        x_support, query_outputs_supports = self.encoder(x_support)
        prototypes = torch.mean(x_support.view(args.train_num_ways, args.num_shots, -1), dim=1)
        prototypes = F.normalize(prototypes, dim=-1).unsqueeze(0) # 1 5 dim
        
        #x_query, query_outputs_queries = self.encoder(x_query, prototypes)
        x_query, query_outputs_queries = self.encoder(x_query)
        support_size = x_support.size(0)
        query_size = x_query.size(0)
        
        included_features_supports = [F.normalize(x_support, dim=-1).unsqueeze(1)]
        included_features_queries = [F.normalize(x_query, dim=-1).unsqueeze(1)]
        '''
        #weighted sum
        included_features_supports = []
        included_features_queries = []
        '''
        
        for s, q in zip(query_outputs_supports, query_outputs_queries):
            included_features_supports.append(F.normalize(s.mean(dim=1), dim=-1).unsqueeze(1)) # b 1 dim
            included_features_queries.append(F.normalize(q.mean(dim=1), dim=-1).unsqueeze(1))
        
        included_features_supports = torch.cat(included_features_supports, dim=1) # b 12 dim
        included_features_queries = torch.cat(included_features_queries, dim=1)
        
        '''
        # weighted sum
        layer_weight = F.softmax(self.layer_weight)
        included_features_supports = F.normalize((included_features_supports * layer_weight.view(1, self.encoder.depth, 1)).sum(dim=1), dim=-1)
        included_features_queries = F.normalize((included_features_queries * layer_weight.view(1, self.encoder.depth, 1)).sum(dim=1), dim=-1)
        
        included_features_supports = torch.cat((F.normalize(x_support, dim=-1), included_features_supports), dim=1)
        included_features_queries = torch.cat((F.normalize(x_query, dim=-1), included_features_queries), dim=1)
        '''
        included_features_supports = included_features_supports.reshape(support_size, -1) # B 12*dim
        included_features_queries = included_features_queries.reshape(query_size, -1)# B 12*dim
        
        prototypes = torch.mean(included_features_supports.view(args.train_num_ways, args.num_shots, -1), dim=1)
        prototypes = F.normalize(prototypes, dim=-1)
        x_query = F.normalize(included_features_queries, dim=-1)
        
        distance = torch.einsum('qd, wd -> qw', x_query, prototypes) # 75 5
        
        return distance
    
    
    def load_backbone(self, patch_size=16, num_query_tokens=1, **kwargs):
        encoder =  VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), attn_class=VQTAttention, num_query_tokens=num_query_tokens, **kwargs)
        
        return encoder
        
    def forward(self, inputs, labels, args, device):
        tasks = split_support_query_set(inputs, labels, device, num_tasks=1, num_class=args.train_num_ways, num_shots=args.num_shots)

        loss = 0
        for x_support, x_query, y_support, y_query in tasks:
            distance = self.calculate_distance(args, x_support, x_query)
            logits = (distance / args.temperature).reshape(-1, args.test_num_ways)
            loss += F.cross_entropy(logits, y_query)
            
        return loss
    
    def finetuning(self, loader, device, n_iters, args):
        correct = 0
        total = 0
        loss = 0
        
        for data in tqdm(loader, desc="Test ..."):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            tasks = split_support_query_set(inputs, labels, device, num_tasks=1, num_class=args.train_num_ways, num_shots=args.num_shots)

            net = copy.deepcopy(self)
            optimizer = optim.SGD([
                {'params': [net.encoder.st_query_tokens], 'lr': 0.1},
                {'params': list(net.head.parameters()), 'lr': 0.1}
            ], momentum=0.9, weight_decay=0.001)
            loss = 0
            
            for x_support, x_query, y_support, y_query in tasks:
                support_size = x_support.size(0)
                query_size = x_query.size(0)
                
                net.train()
                for _ in range(n_iters):
                    x_support_, query_outputs_supports = net.encoder(x_support)
                    included_features_supports = [F.normalize(x_support_, dim=-1).unsqueeze(1)]
                    
                    for s in query_outputs_supports:
                        included_features_supports.append(F.normalize(s.mean(dim=1), dim=-1).unsqueeze(1)) # b 1 dim
                        
                    included_features_supports = torch.cat(included_features_supports, dim=1) 
                    included_features_supports = included_features_supports.reshape(support_size, -1)
                    logits = net.head(included_features_supports)
                    loss = F.cross_entropy(logits, y_support)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                net.eval()

                x_query, query_outputs_queries = net.encoder(x_query)
                included_features_queries = [F.normalize(x_query, dim=-1).unsqueeze(1)]
                
                for s in query_outputs_queries:
                    included_features_queries.append(F.normalize(s.mean(dim=1), dim=-1).unsqueeze(1)) # b 1 dim
                    
                included_features_queries = torch.cat(included_features_queries, dim=1) # b 12 dim
                included_features_queries = included_features_queries.reshape(query_size, -1)
                logits = net.head(included_features_queries)
                
                _, predicted = torch.max(logits.data, 1)
                correct += (predicted == y_query).sum().item()
                total += y_query.size(0)
        acc = 100 * correct / total
            
        return acc
    
    
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