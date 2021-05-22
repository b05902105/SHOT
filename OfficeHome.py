import numpy as np
from PIL import Image
import argparse
import random
from argparse import Namespace
from itertools import groupby

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms

from object.loss import CrossEntropyLabelSmooth, Entropy
from object import network
import os, sys
import os.path as osp

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

class ImageList(Dataset):
    def __init__(self, imgs_path, transform, mode='RGB'):
        self.imgs_path = imgs_path
        self.transform = transform
        self.mode = mode
    def __len__(self):
        return len(self.imgs_path)
    def __getitem__(self, idx):
        path, label = self.imgs_path[idx].split(',')
        img = Image.open(path).convert(self.mode)
        return self.transform(img), int(label)

def train_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    ])

def test_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    ])


def load_data(source_path, target_path, args):
    margs, sargs = args
    dsets = {}
    dloaders = {}

    src_txt = open(source_path, 'r').readlines()
    target_txt = open(target_path, 'r').readlines()

    if margs.imbalance_ratio > 0:
        target_txt = imbalance_sampler(target_txt, margs.imbalance_ratio)

    dsize = len(src_txt)
    train_size = int(0.9 * dsize)

    train_txt, val_txt = torch.utils.data.random_split(src_txt, [train_size, dsize - train_size])

    bsize = sargs.batch_size
    dsets['source_train'] = ImageList(train_txt, transform=train_transform())
    dloaders['source_train'] = DataLoader(dsets['source_train'], batch_size=bsize, shuffle=True, drop_last=False, num_workers=4, pin_memory=True)

    dsets['source_val'] = ImageList(val_txt, transform=test_transform())
    dloaders['source_val'] = DataLoader(dsets['source_val'], batch_size=bsize, shuffle=True, drop_last=False, num_workers=4, pin_memory=True)

    dsets['target_train'] = ImageList(target_txt, transform=train_transform())
    dloaders['target_train'] = DataLoader(dsets['target_train'], batch_size=bsize*2, shuffle=True, drop_last=False, num_workers=4, pin_memory=True)

    dsets['target_test'] = ImageList(target_txt, transform=test_transform())
    dloaders['target_test'] = DataLoader(dsets['target_test'], batch_size=bsize*2, shuffle=True, drop_last=False, num_workers=4, pin_memory=True)

    return dsets, dloaders

def cal_acc(loader, model):
    model.eval()
    pred, true = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.cuda()
            output, _ = model.forward(x)
            pred.append(output.float().cpu())
            true.append(y.float())

    pred, true = torch.cat(pred), torch.cat(true)
    pred = nn.Softmax(dim=1)(pred)
    _, pred = torch.max(pred, 1)
    acc = (torch.squeeze(pred).float() == true).float().mean()
    return acc.item()

def source_train(dloaders, margs, sargs):
    param_group = []
    learning_rate = 1e-2

    model = Model(margs, sargs)
    best_model = Model(margs, sargs)

    for k, v in model.F.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate*0.1}]
    for k, v in model.B.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in model.C.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = 20 * len(dloaders['source_train'])

    best_acc = 0

    model.train()

    for iter_num in range(max_iter):
        total_loss = 0
        total_length = 0
        for i, (source_x, source_y) in enumerate(dloaders['source_train']):
            lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
            source_x, source_y = source_x.cuda(), source_y.cuda()

            outputs, _ = model.forward(source_x)
            loss = CrossEntropyLabelSmooth(num_classes=65, epsilon=0.1)(outputs, source_y)

            total_loss += len(source_x)*loss.item()
            total_length += len(source_x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Step: %02d/%02d, Training Loss: %.4f' % (i+1, len(dloaders['source_train']), total_loss / total_length), end='\r')

        acc_val = cal_acc(dloaders['source_val'], model)
        model.train()
        print('Iter: %03d/%03d, Valid Acc: %.2f%%' % (iter_num + 1, max_iter, 100*acc_val))

        if acc_val > best_acc:
            best_acc = acc_val
            best_model.copy(model)

    best_model.save(source=True)

def target_train(dloaders, model):
    model.target_train_mode()

    for k, v in model.C.named_parameters():
        v.requires_grad = False

    param_group = []
    learning_rate = 1e-2
    for k, v in model.F.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate*0.1}]
    for k, v in model.B.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = 50

    for iter_num in range(max_iter):
        for i, (target_x, target_y) in enumerate(dloaders['target_train']):
            lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
            target_x, target_y = target_x.cuda(), target_y.cuda()

            output, features = model.forward(target_x)

            softmax_output = nn.Softmax(dim=1)(output)
            entropy_loss = torch.mean(Entropy(softmax_output))

            if model.margs.info_max:
                msoftmax = softmax_output.mean(dim=0)
                entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))

            optimizer.zero_grad()
            entropy_loss.backward()
            optimizer.step()

            print('Iter: %02d, Step: %02d/%02d' % (iter_num+1, i+1, len(dloaders['target_train'])), end='\r')

    model.save(source=False)

def gen_path(path, name):
    res = ''
    path = osp.join(path, name)
    for i, sub_forder in enumerate(sorted(os.listdir(path))):
        for file in sorted(os.listdir(osp.join(path, sub_forder))):
            res += osp.join(path, sub_forder, file) + ',%d\n' % (i)

    return res

def arguments_parsing():
    parser = argparse.ArgumentParser()

    model = parser.add_argument_group('model')

    model.add_argument('-im', '--info_max', action='store_true')
    model.add_argument('-s', '--source', required=False, type=int, default=0)
    model.add_argument('-t', '--target', required=False, type=int, default=1)
    model.add_argument('-imr', '--imbalance_ratio', required=False, type=float, default=1)

    sys = parser.add_argument_group('sys')
    sys.add_argument('-m', '--mode', choices=['source_train', 'target_train', 'target_test'], required=True)
    sys.add_argument('-mp', '--model_path', default='./model/OfficeHome')
    sys.add_argument('-dp', '--data_path', default='/tmp2/yc980802/da/data/OfficeHome')
    sys.add_argument('-sm', '--source_model', action='store_true')
    sys.add_argument('-bs', '--batch_size', type=int, default=32)

    args = parser.parse_args()
    model = Namespace(**{a.dest:args.__dict__[a.dest] for a in model._group_actions})
    sys = Namespace(**{a.dest:args.__dict__[a.dest] for a in sys._group_actions})

    return model, sys

class Model:
    def __init__(self, margs, sargs):
        self.F = network.ResBase(res_name='resnet50').cuda()
        self.B = network.feat_bootleneck(type='bn', feature_dim=self.F.in_features, bottleneck_dim=256).cuda()
        self.C = network.feat_classifier(type='wn', class_num=65, bottleneck_dim=256).cuda()
        self.margs = margs
        self.sargs = sargs
    def save(self, source=True):
        argstr = str({'source': margs.source}) if source else str(vars(margs))
        path = osp.join(self.sargs.model_path, argstr)
        os.mkdir(path)
        torch.save(self.F.state_dict(), osp.join(path, 'F.pt'))
        torch.save(self.B.state_dict(), osp.join(path, 'B.pt'))
        torch.save(self.C.state_dict(), osp.join(path, 'C.pt'))
    def load(self, source=True):
        argstr = str({'source': margs.source}) if source else str(vars(margs))
        path = osp.join(self.sargs.model_path, argstr)
        self.F.load_state_dict(torch.load(osp.join(path, 'F.pt')))
        self.B.load_state_dict(torch.load(osp.join(path, 'B.pt')))
        self.C.load_state_dict(torch.load(osp.join(path, 'C.pt')))

    def target_train_mode(self):
        self.F.train()
        self.B.train()
        self.C.eval()

    def train(self):
        self.F.train()
        self.B.train()
        self.C.train()

    def eval(self):
        self.F.eval()
        self.B.eval()
        self.C.eval()

    def forward(self, x):
        feature = self.B(self.F(x))
        return self.C(feature), feature

    def copy(self, m):
        self.F, self.B, self.C = m.F, m.B, m.C

def imbalance_sampler(data, imr=1):
    group = [list(g) for k, g in groupby(data, key=lambda a: int(a[-2]))]

#     sampling_g = [np.random.choice(g, size=int(len(g)*imr), replace=False) for g in group]
#     l = [len(x) for x in sampling_g]
#     ret = [item for g in sampling_g for item in g]

#     return ret

    cls_num = len(group)
    cls_max = min([len(g) for g in group])
    random.shuffle(group)
    sampling_g = [np.random.choice(g, size=int(cls_max*imr**(i/(cls_num-1)))) for i, g in enumerate(group)]
    ret = [item for g in sampling_g for item in g]
    return ret

if __name__ == '__main__':
    margs, sargs = arguments_parsing()
    names = ['Art', 'Clipart', 'Product', 'RealWorld']

    class_num = 65

    source_path = osp.join(sargs.data_path, names[margs.source] + '.txt')
    target_path = osp.join(sargs.data_path, names[margs.target] + '.txt')

    dsets, dloaders = load_data(source_path, target_path, (margs, sargs))
    model = Model(margs, sargs)

    if sargs.mode == 'source_train':
        source_train(dloaders, margs, sargs)
    elif sargs.mode == 'target_train':
        model.load(source=True)
        target_train(dloaders, model)
    else:
        model.load(source=sargs.source_model)
        print('Accuracy: %.2f%%' % (100*cal_acc(dloaders['target_test'], model)))
