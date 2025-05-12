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
import torchvision.transforms as transforms
import kornia.augmentation as K

from utils import split_support_query_set
import backbone.vision_transformer as vit
from test_models.baseline import Baseline

def get_augmentation(name, input_shape):
    if name == 'none':
        return K.AugmentationSequential(
            #return_transform=False, # `return_transform` is deprecated. Please access `.transform_matrix` in `AugmentationSequential` instead.
            same_on_batch=False,
        )

    elif name == 'weak':
        aug = K.AugmentationSequential(
            K.RandomCrop(input_shape[1:], padding=4, padding_mode='reflect', resample='BICUBIC'),
            K.RandomHorizontalFlip(),
            #return_transform=False,
            same_on_batch=False,
        )
        return aug

    elif name == 'strong':
        aug = K.AugmentationSequential(
            K.RandomResizedCrop(input_shape[1:], scale=(0.08, 1.0), resample='BICUBIC'),
            K.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            K.RandomGrayscale(p=0.2),
            K.RandomHorizontalFlip(),
            #return_transform=False,
            same_on_batch=False,
        )
        return aug

    else:
        raise Exception(f'Unknown Augmentation: {name}')


class VisionTransformer(vit.VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def prepare_tokens(self, x, ntoken=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding
            
        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)
        
        return self.pos_drop(x)
    
    def forward(self, x, ntoken=None):
        x = self.prepare_tokens(x, ntoken)
        B = x.size(0)
        if ntoken != None:
            T = ntoken.size(1)
            ntoken = ntoken.repeat(B, 1, 1)
            x = torch.cat((ntoken, x), dim=1)
            
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        if ntoken != None:
            return x[:, :T].mean(dim=1)
        return x[:, 0]


class TTTSSL(nn.Module):
    def __init__(self, img_size, patch_size):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.encoder = self.load_backbone()
        self.ntoken = nn.Parameter(torch.zeros(1, 1, self.encoder.num_features))
        self.fix_encoder()
        self.strong_aug = get_augmentation('strong', (3, 224, 224))
        self.weak_aug = get_augmentation('weak', (3, 224, 224))
        
    def load_backbone(self, patch_size=16, **kwargs):
        encoder =  VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        
        return encoder

    def fix_encoder(self):
        for name, param in self.encoder.named_parameters():
            if 'norm' in name or 'st_' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def inter_class_distance_loss(self, prototypes):
        n_way = prototypes.size(0)
        loss = 0
        for i in range(n_way):
            for j in range(i+1, n_way):
                dist = F.pairwise_distance(prototypes[i].unsqueeze(0), prototypes[j].unsqueeze(0), p=2)
                loss += -dist
        return loss / (n_way * (n_way - 1) / 2)
    
    def calculate_logits(self, args, x_support, x_query, ntoken):
        x_support = self.encoder(x_support, ntoken)
        prototypes = torch.mean(x_support.view(args.train_num_ways, args.num_shots, -1), dim=1)
        prototypes = F.normalize(prototypes, dim=-1)
        inter_class_loss = self.inter_class_distance_loss(prototypes)
        
        x_query = self.encoder(x_query, ntoken)
        x_query = F.normalize(x_query, dim=-1)
        
        logits = torch.einsum('qd, wd -> qw', x_query, prototypes) # 75 5
        
        return logits, inter_class_loss
    
    def forward(self, inputs, labels, args, device):
        tasks = split_support_query_set(inputs, labels, device, num_tasks=1, num_class=args.train_num_ways, num_shots=args.num_shots)
        
        total_loss = 0
        for x_support, x_query, y_support, y_query in tasks:
            logits1 = self.calculate_logits(args, self.strong_aug(x_support), self.strong_aug(x_query), self.ntoken)
            logits2 = self.calculate_logits(args, self.weak_aug(x_support), self.weak_aug(x_query), self.ntoken)
            
            loss = (self.cross_entropyloss(logits1, logits2) + self.cross_entropyloss(logits2, logits1)) / 2
            loss += (F.cross_entropy(logits1, y_query) + F.cross_entropy(logits2, y_query)) / 2
            total_loss += loss
        
        return total_loss
    
    def cross_entropyloss(self, logits1, logits2):
        logits2 = logits2.detach()
        p1_log = F.log_softmax(logits1 / 0.1, dim=1)
        p2 = F.softmax(logits2 / 0.04, dim=1)
        
        loss = - (p2 * p1_log).sum(dim=1).mean()
        return loss
    
    def fewshot_acc(self, args, inputs, labels, device):
        tasks = split_support_query_set(inputs, labels, device, num_tasks=1, num_class=args.train_num_ways, num_shots=args.num_shots)
        
        query_size = args.num_shots * args.num_queries
        n_prompt = 1
        #ntoken = nn.Parameter(torch.zeros(1, n_prompt, self.encoder.num_features, device=device))
        ntoken = copy.deepcopy(self.ntoken)
                
        optimizer = optim.AdamW([
            {'params': [ntoken], 'lr': 1e-6},
        ])
        
        correct = 0
        loss = 0
        total = 0
        n_iters = 10
        
        for x_support, x_query, y_support, y_query in tasks:
            self.train()
            
            for i in range(n_iters):
                logits1, inter_cls_loss1 = self.calculate_logits(args, self.strong_aug(x_support), self.strong_aug(x_query), ntoken)
                logits2, inter_cls_loss2 = self.calculate_logits(args, self.weak_aug(x_support), self.weak_aug(x_query), ntoken)
                
                loss = (self.cross_entropyloss(logits1, logits2) + self.cross_entropyloss(logits2, logits1)) / 2
                
                if i == 0:
                    first_loss = loss
                elif i == n_iters-1:
                    last_loss = loss
                    print(first_loss - last_loss)
                    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            self.eval()
            
            
            with torch.no_grad():
                distance, _ = self.calculate_logits(args, x_support, x_query, ntoken)
                logits = (distance / args.temperature).reshape(-1, args.test_num_ways)
                _, predicted = torch.max(logits.data, 1)
                correct += (predicted == y_query).sum().item()
                total += y_query.size(0)
        
            acc = 100 * correct / total
            #print(acc)
        return acc