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
from test_models.cls_modification import CLSMOD

class CLSMODSSL(CLSMOD):
    def __init__(self, img_size, patch_size, finetune_norm, permute_pos):
        super(CLSMODSSL, self).__init__(img_size, patch_size, finetune_norm, permute_pos)
        self.img_size = img_size
        self.patch_size = patch_size
        self.encoder = self.load_backbone()
        self.encoder_k = self.load_backbone()
        
        self.projector = self.make_mlp(self.encoder.num_features)
        self.projector_k = self.make_mlp(self.encoder.num_features)
        self.predictor = self.make_mlp(input_dim=128, last_bn=False)

        self.dim = 128
        
        q_size=16384
        momentum=0.99
        
        self.register_buffer('queue', F.normalize(torch.randn(q_size, self.dim), p=2))
        self.register_buffer('idx', torch.zeros(1, dtype=torch.int64))
        self.max_queue_size = q_size
        self.momentum = momentum
        self.shots = 4
        
        for k_param, q_param in zip(self.encoder_k.parameters(), self.encoder.parameters()):
            k_param.data.copy_(q_param.data)
            k_param.requires_grad = False
            
        for k_param, q_param in zip(self.projector_k.parameters(), self.projector.parameters()):
            k_param.data.copy_(q_param.data)
            k_param.requires_grad = False
                
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
    
    def forward(self, inputs, labels, args, device):
        inputs[0], inputs[1] = inputs[0].to(device), inputs[1].to(device)
        labels  = labels.to(device)
        
        batch_size = inputs[0].size(0)
        
        x_q = inputs[0]
        x_k = inputs[1]
        
        #momentum update
        with torch.no_grad():
            for k_param, q_param in zip(self.encoder_k.parameters(), self.encoder.parameters()):
                k_param.data = self.momentum * k_param.data + (1 - self.momentum) * q_param.data
            
            for k_param, q_param in zip(self.projector_k.parameters(), self.projector.parameters()):
                k_param.data = self.momentum * k_param.data + (1 - self.momentum) * q_param.data
                
        q = F.normalize(self.predictor(self.projector(self.encoder(x_q))))
        
        #psco loss
        with torch.no_grad():
            k = F.normalize(self.projector_k(self.encoder_k(x_k))).detach()
            sim = torch.einsum('bd, qd -> bq', k, self.queue.clone().detach())
            labels_tilde = sinkhorn(sim, device)
            support_set, labels = self.select_topk(self.queue, labels_tilde)
            labels = labels.to(device)
        
        logits = torch.einsum('bd, sd -> bs', q, support_set)
        loss_psco = logits.logsumexp(dim=1) - (torch.sum(logits * labels, dim=1) / self.shots)
        loss_psco = loss_psco.mean()
        
        logits_moco_positive = torch.einsum('bd, bd -> b', q, k).unsqueeze(1) # b 1
        logits_moco_negative = torch.einsum('bd, qd -> bq', q, self.queue.clone().detach())
        logits_moco = torch.cat((logits_moco_positive, logits_moco_negative), dim=1) # b 1+q
        
        labels_moco = torch.zeros(batch_size).long().to(device)
        loss_moco = F.cross_entropy(logits_moco / 0.2, labels_moco)
        
        #enqueue
        self.enqueue(batch_size, k)
        
        return loss_psco + loss_moco
    
    def select_topk(self, queue, labels_tilde):
        _, indicies = torch.topk(labels_tilde, k=self.shots, dim=1)
        b, k = indicies.shape
        support_set = queue[indicies].clone().detach().view(b * k, -1) # b k d -> (bk) d
        
        labels = torch.zeros([b, b * k])
        
        for i in range(b):
            labels[i, k * i : k * i + k] = 1
        
        return support_set, labels
    
    def enqueue(self, batch_size, k):
        with torch.no_grad():
            idx = int(self.idx)
            self.queue[idx : idx + batch_size] = k
            self.idx[0] = (idx + batch_size) % self.max_queue_size

def sinkhorn(scores, device, eps=0.05, niters=3):
    code = torch.transpose(torch.exp(scores / eps), 0, 1)
    code /= torch.sum(code)
    
    k, b = code.shape
    u, r, c = torch.zeros(k), torch.ones(k) / k, torch.ones(b) / b
    u, r, c = u.to(device), r.to(device), c.to(device)
    
    for _ in range(niters):
        u = torch.sum(code, dim=1)
        code *= (r / u).unsqueeze(1) # k * b
        code *= (c / torch.sum(code, dim=0)).unsqueeze(0) # k * b
    
    return torch.transpose((code / torch.sum(code, dim=0, keepdim=True)), 0, 1) # b * k

