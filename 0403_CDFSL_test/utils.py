import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch.optim as optim


def parsing_argument():
    parser = argparse.ArgumentParser(description="argparse_test")

    parser.add_argument('-d', '--dataset', metavar='str', type=str, help='dataset [miniimagenet, cifarfs]', default='miniimagenet')
    parser.add_argument('-opt', '--optimizer', metavar='str', type=str, help='optimizer [adam, sgd]', default='sgd')
    parser.add_argument('-crt', '--criterion', metavar='str', type=str, help='criterion [ce, mse]', default='ce')
    parser.add_argument('-tr', '--train', help='train', action='store_true')
    parser.add_argument('-val', '--val', help='validation', action='store_true')
    parser.add_argument('-ad', '--adaptation', help='adaptation', action='store_true')
    parser.add_argument('-m', '--model', metavar='str', type=str, help='models [protonet, feat, relationnet]', default='moco')
    parser.add_argument('-bs', '--batch_size', metavar='int', type=int, help='batchsize', default=256)
    parser.add_argument('-tc', '--test', metavar='str', type=str, help='knn, fewshot, cross-domain', default='knn')
    parser.add_argument('-b', '--backbone', metavar='str', type=str, help='conv5, resnet10|18', default='resnet10')
    parser.add_argument('-mixup', '--mixup', help='mixup in psco', action='store_true')
    parser.add_argument('-log', '--log', metavar='str', type=str, help='log', default='')
    
    # for scheduling
    parser.add_argument('-e', '--epochs', metavar='int', type=int, help='epochs', default=2)
    parser.add_argument('-lr', '--learningrate', metavar='float', type=float, help='lr', default=0.01)
    parser.add_argument('-warmup', '--warmup', metavar='float', type=float, help='warmupepochs', default=0)
    
    # for fewshot
    parser.add_argument('-tr_ways', '--train_num_ways', metavar='int', type=int, help='ways', default=5)
    parser.add_argument('-ts_ways', '--test_num_ways', metavar='int', type=int, help='ways', default=5)
    parser.add_argument('-shots', '--num_shots', metavar='int', type=int, help='shots', default=5)
    parser.add_argument('-tasks', '--num_tasks', metavar='int', type=int, help='tasks', default=1)
    parser.add_argument('-q', '--num_queries', metavar='int', type=int, help='queries', default=15)
    parser.add_argument('-ep', '--episodes', metavar='int', type=int, help='episodes', default=100)
    
    # for ViT
    parser.add_argument('-img_size', '--img_size', metavar='int', type=int, help='input img size', default=84)
    parser.add_argument('-patch_size', '--patch_size', metavar='int', type=int, help='patch size', default=6)
    
    # for setfsl
    parser.add_argument('-num_g_prompts', '--num_g_prompts', metavar='int', type=int, help='patch size', default=0)
    parser.add_argument('-num_e_prompts', '--num_e_prompts', metavar='int', type=int, help='patch size', default=0)
    parser.add_argument('-temperature', '--temperature', metavar='float', type=float, help='patch size', default=0.001)
    parser.add_argument('-prompt', '--prompt', metavar='str', type=str, help='prefix, prompt', default='prompt')
    
    parser.add_argument('-num_objects', '--num_objects', metavar='int', type=int, default=2)
    parser.add_argument('-layer', '--layer', metavar='int', type=int, default=11)
    parser.add_argument('-withcls', '--withcls', action='store_true')
    parser.add_argument('-continual_layers', '--continual_layers', nargs='+', type=int, default=None)
    
    parser.add_argument('-train_w_qkv', '--train_w_qkv', action='store_true')
    parser.add_argument('-train_w_o', '--train_w_o', action='store_true')
    
    parser.add_argument('-is_new_epoch', '--is_new_epoch', action='store_true')
    
    parser.add_argument('-checkpointdir', '--checkpointdir', metavar='str', type=str, default='dino_deitsmall16_pretrain.pth')
    
    return parser.parse_args()

def set_parameters(args, net, len_trainloader):
    if args.optimizer == 'adamW':
        sourcetrain_param = []
        encoder_param = []
        for name, param in net.named_parameters():
            if 'encoder' not in name or 'st_' in name:
                print(name)
                sourcetrain_param.append(param)
            else:
                encoder_param.append(param)

        optimizer = optim.AdamW([{'params':sourcetrain_param, 'lr':args.learningrate}, {'params':encoder_param, 'lr':1e-6}])
        #optimizer = optim.AdamW(params=net.parameters(), lr=args.learningrate, weight_decay=0.0005)
        #scheduler = WarmupCosineAnnealingScheduler(optimizer, warmup_steps=args.warmup, base_lr=args.learningrate, T_max=args.epochs, eta_min=1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-6)
        #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[9999], gamma=0.1)
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(params=net.parameters(), lr=args.learningrate, weight_decay=0.0005, momentum=0.9)
        #scheduler = WarmupCosineAnnealingScheduler(optimizer, warmup_steps=10, base_lr=args.learningrate, T_max=100, eta_min=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    return optimizer,scheduler



def split_support_query_set(x, y, device, num_class=5, num_shots=5, num_queries=15, num_tasks=1, training=False):
    
    x_list = torch.chunk(x, num_tasks)
    y_list = torch.chunk(y, num_tasks)
    tasks = []
    
    for i in range(num_tasks):
        x, y = x_list[i], y_list[i]
        num_sample_support = num_class * num_shots
        x_support, x_query = x[:num_sample_support], x[num_sample_support:]
        y_support, y_query = y[:num_sample_support], y[num_sample_support:]
        
        _classes = torch.unique(y_support)
        support_idx = torch.stack(list(map(lambda c: y_support.eq(c).nonzero(as_tuple=False).squeeze(1), _classes)))
        xs = torch.cat([x_support[idx_list] for idx_list in support_idx])
        
        query_idx = torch.stack(list(map(lambda c: y_query.eq(c).nonzero(as_tuple=False).squeeze(1), _classes)))
        xq = torch.cat([x_query[idx_list] for idx_list in query_idx])
        
        ys = torch.arange(0, len(_classes), 1 / num_shots).long().to(device)
        yq = torch.arange(0, len(_classes), 1 / num_queries).long().to(device)
        
        tasks.append([xs, xq, ys, yq])
        
    return tasks


def load_model(args):
    if args.model == 'baseline':
        import test_models.baseline as baseline
        net = baseline.Baseline(args.img_size, args.patch_size)
    elif args.model == 'prompt':
        import test_models.prompt as prompt
        net = prompt.Prompt(args.img_size, args.patch_size)
    elif args.model == 'clsmod':
        import test_models.cls_modification as clsmod
        net = clsmod.CLSMOD(args.img_size, args.patch_size)
    elif args.model == 'posmod':
        import test_models.positonal_modification as posmod
        net = posmod.POSMOD(args.img_size, args.patch_size)
    elif args.model == 'clsmod_pospermute':
        import test_models.clsmod_pospermute as clsmod
        net = clsmod.CLSMOD(args.img_size, args.patch_size)
    elif args.model == 'prototype_contextualization':
        import test_models.prototype_contextualization as prototype_contextualization
        net = prototype_contextualization.SETFSL(args.img_size, args.patch_size, args.num_objects, args.temperature, args.layer, args.withcls, args.continual_layers, args.train_w_qkv, args.train_w_o)
    elif args.model == 'lowlayer_contextualization':
        import test_models.lowlayer_contextualization as lowlayer_contextualization
        net = lowlayer_contextualization.SETFSL(args.img_size, args.patch_size, args.num_objects, args.temperature, args.layer, args.withcls, args.continual_layers, args.train_w_qkv, args.train_w_o)
    return net
