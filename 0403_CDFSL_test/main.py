import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import torch.nn.utils as utils
import torch.optim as optim

import random
import numpy as np

from tqdm import tqdm

import data.dataloader as dataloader

from utils import parsing_argument, load_model, set_parameters, split_support_query_set

from torch.utils.tensorboard import SummaryWriter


def train_per_epoch(args, dataloader, net, optimizer, scheduler, device):
    is_new_epoch = True
    running_loss = 0.0
    print(net.checkencentrop)
    net.train()
    
    representations = None
    label_list = None
    
    for data in dataloader:
        inputs, labels = data
        if not isinstance(inputs, list):
            inputs, labels = inputs.to(device), labels.to(device)
        loss = net(inputs, labels, args, device)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        args.is_new_epoch = True
        
        running_loss += loss.item()
        args.is_new_epoch = False
    
    return running_loss/len(dataloader), representations, label_list

def finetuning(testloader, net, args, device):
    acc = net.finetuning(testloader, device, 50, args)
    return acc

def fewshot_test(testloader, net, args, device):
    total_acc = 0
    acc_lst = []
    
    if args.adaptation:
        accuracy = net.ft_fewshot_acc(testloader, device, n_iters=100, args=args)
    else:
        for data in testloader:
            inputs, labels = data # inputs 100 3(aug) 3(channels) 224 224
            inputs, labels = inputs.to(device), labels.to(device)
            
            net.eval()
            acc = net.fewshot_acc(args, inputs, labels, device)
            #total_acc += acc
            acc_lst.append(acc)
        
        acc_all = np.asarray([a.cpu().item() if torch.is_tensor(a) else a for a in acc_lst])
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        
    #'Acc = %4.2f%% +- %4.2f%%'%(acc_mean, 1.96* acc_std/np.sqrt(iter_num))
    return acc_mean

def crossdomain_test(args, net, device, outputs_log):
    print('--- crossdomain test ---')
    if args.dataset == 'BSCD':
        dataset_list = ['CropDisease','EuroSAT', 'ISIC', 'ChestX', 'miniimagenet']
    elif args.dataset == 'FWT':
        dataset_list = ['cub', 'cars', 'places', 'plantae']
    elif args.dataset == 'all':
        dataset_list = ['CropDisease','EuroSAT', 'ISIC', 'ChestX','cub', 'cars', 'places', 'plantae']
    else:
        print('invalid dataset')
        return
    
    total = 0
    for dataset in dataset_list:
        trainloader, testloader, valloader, num_classes = dataloader.load_dataset(args, dataset)
        print(f'--- {dataset} test ---')
        acc = fewshot_test(testloader, net, args, device=device)
        #acc = finetuning(testloader, net, args, device=device)
        
        total += acc
        print(f'{dataset} fewshot_acc : %.3f'%(acc))
        print(f'{dataset} fewshot_acc : %.3f'%(acc), file=outputs_log)
    mean = total / len(dataset_list)
    print(f'mean_acc : %.3f'%(mean))
    print(f'mean_acc : %.3f'%(mean), file=outputs_log)


def train(args, trainloader, testloader, valloader, net, optimizer, scheduler, device, writer, outputs_log):
    max_acc = 0
    for epoch in range(args.epochs):
        running_loss, representations, label_list = train_per_epoch(args, trainloader, net, optimizer, scheduler, device)
        acc = 0
        if args.test == 'fewshot' and (epoch+1) % 5 == 0 or epoch == 0:
            acc = fewshot_test(valloader, net, args, device=device)
            writer.add_scalar('train / fewshot_acc', acc, epoch+1)
        
        if scheduler:
            scheduler.step()
        
        lr = optimizer.param_groups[0]['lr']
        
        print('epoch[%d] - training loss : %.3f / %s_acc : %.3f / lr : %f'%(epoch+1, running_loss, args.test, acc, lr))
        print('epoch[%d] - training loss : %.3f / %s_acc : %.3f'%(epoch+1, running_loss, args.test, acc), file=outputs_log)
        writer.add_scalar('train / train_loss', running_loss, epoch+1)
        writer.add_scalar('train / learning_rate', lr, epoch+1)
        
        torch.save(net.state_dict(), f'./{args.model}_{args.epochs}ep_{args.learningrate}lr_{args.log}.pt')

        if acc > max_acc:
            max_acc = acc
            torch.save(net.state_dict(), f'./{args.model}_best_ep_{args.learningrate}lr_{args.log}.pt')
            
        running_loss = 0.0
        
    print('Training finished',file=outputs_log)


def main():
    seed = 0
    print("set seed = %d" % seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    args = parsing_argument()
    cur_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    
    if args.train:
        outputs_log = open(f'./outputs/{args.model}_{args.epochs}ep_{args.learningrate}lr_{args.log}_{cur_time}.txt','w')
        writer = SummaryWriter(f'./logs/{args.model}_{args.epochs}ep_{args.learningrate}lr_{args.log}_{cur_time}')
    elif args.test:
        outputs_log = open(f'./outputs/{args.model}_test_{args.dataset}_{cur_time}_{args.log}.txt','w')
        writer = None
    
    net = load_model(args)
    net.to(device)
    
    trainable_parameters = sum(p.numel() for p in net.encoder.parameters() if p.requires_grad)
    total_parameters = sum(p.numel() for p in net.encoder.parameters())
    print(trainable_parameters, total_parameters)
    
    if args.train:
        net.encoder.load_state_dict(torch.load('dino_deitsmall16_pretrain.pth'), strict=False)
        trainloader, testloader, valloader, num_classes = dataloader.load_dataset(args, args.dataset)
        optimizer,scheduler = set_parameters(args, net, len(trainloader))
        print(f"--- train ---")
        train(args, trainloader, testloader, valloader, net, optimizer, scheduler, device, writer, outputs_log)
    
    if args.test == 'fewshot':
        #net.encoder.load_state_dict(torch.load('dino_deitsmall16_pretrain.pth'), strict=False)
        trainloader, testloader, valloader, num_classes = dataloader.load_dataset(args, args.dataset)
        print(f'--- {args.dataset} test ---')
        acc = fewshot_test(testloader, net, args, device)
        print('fewshot_acc : %.3f'%(acc))
        print('fewshot_acc : %.3f'%(acc), file=outputs_log)
    
    elif args.test == 'crossdomain':
        #net.encoder.load_state_dict(torch.load('dino_deitsmall16_pretrain.pth'), strict=False)
        net.load_state_dict(torch.load(args.checkpointdir), strict=False)
        
        '''
        for name, param in net.named_parameters():
            if 'alpha' in name:
                print(name, torch.sigmoid(param))
        '''
                
        print(args.checkpointdir)
        crossdomain_test(args, net, device, outputs_log)
        
    outputs_log.close()
    if writer != None:
        writer.close()

if __name__ == "__main__":
    main()
