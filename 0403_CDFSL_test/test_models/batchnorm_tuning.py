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


class CustomBatchNorm(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.Tensor([0.]))
        #self.beta = nn.Parameter(torch.Tensor([0.]))

    def forward(self, x, x_mean=None, x_var=None, weight=1., bias=0., alpha=0.5, beta=0.5):
        batch_mean, batch_var = None, None
        alpha = torch.sigmoid(self.alpha)
        #beta = torch.sigmoid(self.beta)
        # supports
        if x_mean == None:
            batch_mean = torch.mean(x, dim=(0,2), keepdim=True) # 1 N 1
            instance_mean = torch.mean(x, dim=-1, keepdim=True)
            mean = alpha * batch_mean + (1 - alpha) * instance_mean
        if x_var == None:
            batch_var = torch.var(x, dim=(0,2), unbiased=False, keepdim=True)
            instance_var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
            var = alpha * batch_var + (1 - alpha) * instance_var + alpha * (1-alpha) * ((batch_mean-instance_mean)**2)
        # queries
        if x_mean != None:
            instance_mean = torch.mean(x, dim=-1, keepdim=True)
            mean = alpha * x_mean + (1 - alpha) * instance_mean
        if x_var != None:
            instance_var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
            var = alpha * x_var + (1 - alpha) * instance_var + alpha * (1-alpha) * ((x_mean-instance_mean)**2)
        
        x_norm = (x - mean) / ((var + self.eps) ** 0.5)
        out = weight * x_norm + bias
        
        if batch_mean!=None and batch_var!=None:
            return out, batch_mean, batch_var
        else:
            return out, mean, var

class BNBlock(vit.Block):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_class=vit.Attention):
        super().__init__(dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                 drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer, attn_class=attn_class)
        self.st_bn1 = CustomBatchNorm()
        self.st_bn2 = CustomBatchNorm()
        
    
    def forward(self, x, return_attention=False, supports_stat=None):
        mean1, var1, mean2, var2 = None, None, None, None
        if supports_stat != None:
            mean1, var1, mean2, var2 = supports_stat[0], supports_stat[1], supports_stat[2], supports_stat[3]

        x_norm, mean1, var1 = self.st_bn1(x, x_mean=mean1, x_var=var1, weight=self.norm1.weight, bias=self.norm1.bias)
        #x_norm = self.norm1(x)
        y, attn = self.attn(x_norm)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x_norm, mean2, var2 = self.st_bn2(x, x_mean=mean2, x_var=var2, weight=self.norm2.weight, bias=self.norm2.bias)
        #x_norm = self.norm2(x)
        x = x + self.drop_path(self.mlp(x_norm))
        
        return x, mean1, var1, mean2, var2
    

class VisionTransformer(vit.VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.st_clsmod = nn.Parameter(torch.zeros_like(self.cls_token))
        self.supports_stat = [None for _ in range(len(self.blocks)+1)]
        self.print_flag = True
        self.st_norm = CustomBatchNorm()
    
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
    
    def forward(self, x, supports_stat=False):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if supports_stat: # queries forward
                assert self.supports_stat[i] != None, 'supports stat is empty'
                #x, mean1, var1, mean2, var2 = blk(x)
                x, mean1, var1, mean2, var2 = blk(x, supports_stat = self.supports_stat[i])
                '''
                if self.print_flag:
                    print(mean1.sum(), var1.sum())
                    print(mean1.squeeze(), var1.squeeze())
                    self.print_flag = False
                '''
            else: # supports forward
                x, mean1, var1, mean2, var2 = blk(x)
                self.supports_stat[i] = [mean1, var1, mean2, var2]
        
        if supports_stat: # queries
            x, mean, var = self.st_norm(x, x_mean=self.supports_stat[-1][0], x_var=self.supports_stat[-1][1], weight=self.norm.weight, bias=self.norm.bias)
        else: # supports
            x, mean, var = self.st_norm(x, weight=self.norm.weight, bias=self.norm.bias)
            self.supports_stat[-1] = [mean, var]
        
        return x[:, 0]



class BNTuning(Baseline):
    def __init__(self, img_size, patch_size):
        super(BNTuning, self).__init__(img_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.encoder = self.load_backbone(patch_size)
        self.head = nn.Linear(self.encoder.num_features, 5)
        self.fix_encoder()

    def fix_encoder(self):
        for name, param in self.encoder.named_parameters():
            if 'norm' in name or 'st_' in name or 'bn' in name:
                print(name)
                param.requires_grad = True
            else:
                param.requires_grad = False
            '''
            if 'cls' in name and 'st' not in name:
                print(name)
                param.requires_grad = False
            '''
    
    def load_backbone(self, patch_size=16, **kwargs):
        encoder =  VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), block_class=BNBlock, **kwargs)
        
        return encoder

    def calculate_distance(self, args, x_support, x_query):
        x_support = self.encoder(x_support)
        prototypes = torch.mean(x_support.view(args.train_num_ways, args.num_shots, -1), dim=1)
        prototypes = F.normalize(prototypes, dim=-1)
        
        x_query = self.encoder(x_query, supports_stat=True)
        x_query = F.normalize(x_query, dim=-1)
        
        distance = torch.einsum('qd, wd -> qw', x_query, prototypes) # 75 5
        
        return distance
        
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
    
    def finetuning(self, loader, device, n_iters, args):
        correct = 0
        total = 0
        loss = 0
        
        for data in tqdm(loader, desc="Test ..."):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            tasks = split_support_query_set(inputs, labels, device, num_tasks=1, num_class=args.train_num_ways, num_shots=args.num_shots)

            net = copy.deepcopy(self)
            
            ft_param = []
            for name, param in net.named_parameters():
                if 'alpha' in name or 'norm' in name or 'st_' in name:
                    ft_param.append(param)
                    
            optimizer = optim.AdamW([
                {'params': ft_param, 'lr': 0.01},
                {'params': list(net.head.parameters()), 'lr': 0.01}
            ])
            loss = 0
            
            for x_support, x_query, y_support, y_query in tasks:
                net.train()
                for _ in range(n_iters):
                    x_support_ = net.encoder(x_support)
                    logits = net.head(x_support_)
                    loss = F.cross_entropy(logits, y_support)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                net.eval()
                
                distance = net.calculate_distance(args, x_support, x_query)
                logits = (distance / args.temperature).reshape(-1, args.test_num_ways)
                _, predicted = torch.max(logits.data, 1)
                correct += (predicted == y_query).sum().item()
                total += y_query.size(0)
            
        for name, param in net.named_parameters():
            if 'alpha' in name:
                print(name, torch.sigmoid(param))
        acc = 100 * correct / total
            
        return acc