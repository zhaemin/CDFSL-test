import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Sampler, DataLoader

import random
import numpy as np
import data.cd_dataset as cd_dataset
import data.additional_transforms as additional_transforms

def worker_init_fn(worker_id):
    seed = 0
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)

class FewShotSampler(Sampler):
    def __init__(self, labels, num_ways, num_shots, num_queries, episodes, num_tasks, data_source=None):
        super().__init__(data_source)
        self.labels = labels
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.num_queries = num_queries
        self.episodes = episodes
        self.num_tasks = num_tasks
        
        self.classes, self.counts = torch.unique(self.labels, return_counts=True)
        self.num_classes = len(self.classes)
        self.data_matrix = torch.Tensor(np.empty((len(self.classes), max(self.counts)), dtype=int)*np.nan)
        self.num_per_class = torch.zeros_like(self.classes)
        
        #data_matrix => 해당 class에 맞는 데이터의 index를 저장
        #np.where => nan인 값들이 2차원으로 반환됨 [[nan, nan, ..., nan]]
        
        for data_idx, label in enumerate(labels):
            self.data_matrix[label, np.where(np.isnan(self.data_matrix[label]))[0][0]] = data_idx 
            self.num_per_class[label] += 1
        
        self.valid_classes = [c.item() for c, count in zip(self.classes, self.num_per_class) if count >= self.num_shots+self.num_queries]
        
    def __iter__(self):
        for _ in range(self.episodes):
            tasks = []
            for t in range(self.num_tasks):
                batch_support_set = torch.LongTensor(self.num_ways*self.num_shots)
                batch_query_set = torch.LongTensor(self.num_ways*self.num_queries)
                
                way_indices = torch.randperm(len(self.valid_classes))[:self.num_ways]
                selected_classes = [self.valid_classes[idx] for idx in way_indices]
                
                for i, label in enumerate(selected_classes):
                    slice_for_support = slice(i*self.num_shots, (i+1)*self.num_shots)
                    slice_for_queries = slice(i*self.num_queries, (i+1)*self.num_queries)
                    
                    samples = torch.randperm(self.num_per_class[label])[:self.num_shots+self.num_queries]
                    batch_support_set[slice_for_support] = self.data_matrix[label][samples][:self.num_shots]
                    batch_query_set[slice_for_queries] = self.data_matrix[label][samples][self.num_shots:]
                
                batch = torch.cat((batch_support_set, batch_query_set))
                tasks.append(batch)
            
            batches = torch.cat(tasks)
            yield batches
            
    def __len__(self):
        return self.episodes


def load_dataset(args, dataset):
    
    transform_test = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
    
    if dataset == 'miniimagenet':
        trainloader, testloader, valloader, num_classes = load_source_data(args)
    else:
        trainloader = None
        valloader = None
        num_classes = None
        
        transform_test = transforms.Compose([
            transforms.Resize([int(args.img_size*1.15), int(args.img_size*1.15)]),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
        
        testset = cd_dataset.load_crossdomain_dataset(dataset, transform_test)
            
        testset_labels = torch.LongTensor(testset.targets)
        test_sampler = FewShotSampler(testset_labels, args.test_num_ways, args.num_shots, args.num_queries, 600, num_tasks=1)
        testloader = DataLoader(testset, batch_sampler=test_sampler, pin_memory=True, num_workers=8, worker_init_fn=worker_init_fn)

    
    return trainloader, testloader, valloader, num_classes

def load_source_data(args):
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.img_size),
            additional_transforms.ImageJitter(dict(Brightness=0.4, Contrast=0.4, Color=0.4)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    transform_test = transforms.Compose([
        transforms.Resize([int(args.img_size*1.15), int(args.img_size*1.15)]),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
    
    trainset = torchvision.datasets.ImageFolder(root='../data/fewshotdata/miniimagenet/data/train', transform=transform_train)
    testset = torchvision.datasets.ImageFolder(root='../data/fewshotdata/miniimagenet/data/test', transform=transform_test)
    valset = torchvision.datasets.ImageFolder(root='../data/fewshotdata/miniimagenet/data/val', transform=transform_test)
    num_classes = 64
    
    #valset = cd_dataset.load_crossdomain_dataset('CropDisease', transform_test)
    
    trainset_labels = torch.LongTensor(trainset.targets)
    testset_labels = torch.LongTensor(testset.targets)
    valset_labels = torch.LongTensor(valset.targets)
    
    train_sampler = FewShotSampler(trainset_labels, args.train_num_ways, args.num_shots, args.num_queries, args.episodes, args.num_tasks)
    test_sampler = FewShotSampler(testset_labels, args.test_num_ways, args.num_shots, args.num_queries, 1000, num_tasks=1)
    val_sampler = FewShotSampler(valset_labels, args.test_num_ways, args.num_shots, args.num_queries, 100, num_tasks=1)
    
    trainloader = DataLoader(trainset, batch_sampler=train_sampler, num_workers=10, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(valset, batch_sampler=val_sampler, num_workers=10, pin_memory=True, worker_init_fn=worker_init_fn)

    if args.test == 'fewshot' or args.test == 'crossdomain':
        testloader = DataLoader(testset, batch_sampler=test_sampler, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    else:
        testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=8, worker_init_fn=worker_init_fn)

    
    return trainloader, testloader, valloader, num_classes