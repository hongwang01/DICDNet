#!/usr/bin/env python
# -*- coding:utf-8 -*-
# (TMI 2021)DICDNet: Deep Interpretable Convolutional Dictionary Networkfor Metal Artifact Reduction in CT Images

from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.functional as  F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import time
import numpy as np
from tensorboardX import SummaryWriter
from dicdnet import DICDNet
from torch.utils.data import DataLoader
from dataset import MARTrainDataset
from math import ceil


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="data/train/", help='txt path to training data')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--patchSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=200, help='total number of training epochs')
parser.add_argument('--batchnum', type=int, default=1000, help='the number of batch')
parser.add_argument('--num_M', type=int, default=32, help='the number of feature maps')
parser.add_argument('--num_Q', type=int, default=32, help='the number of channel concatenation')
parser.add_argument('--T', type=int, default=3, help='the number of ResBlocks in every ProxNet')
parser.add_argument('--S', type=int, default=10, help='Stage number')
parser.add_argument('--etaM', type=float, default=1, help='stepsize for updating M')
parser.add_argument('--etaX', type=float, default=5, help='stepsize for updating X')
parser.add_argument('--resume', type=int, default=0, help='continue to train from epoch')
parser.add_argument("--milestone", type=int, default=[30,60,90,120,150,180], nargs='+',
                    help="When to decay learning rate")
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument('--log_dir', default='./logs/', help='tensorboard logs')
parser.add_argument('--model_dir', default='./models/', help='saving model')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--Xl2', type=float, default=1, help='loss weights')
parser.add_argument('--Xl1', type=float, default=5e-4, help='loss weights')
parser.add_argument('--Al1', type=float, default=54-4, help='loss weights')
opt = parser.parse_args()

if opt.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    device = torch.device("cuda:0")

# create path
try:
    os.makedirs(opt.log_dir)
except OSError:
    pass
try:
    os.makedirs(opt.model_dir)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
cudnn.benchmark = True

def train_model(net, optimizer, lr_scheduler, datasets):
    data_loader = DataLoader(datasets, batch_size=opt.batchSize, shuffle=True,num_workers=int(opt.workers), pin_memory=True)
    num_data = len(datasets)
    num_iter_epoch = ceil(num_data / opt.batchSize)
    writer = SummaryWriter(opt.log_dir)
    step = 0
    for epoch in range(opt.resume, opt.niter):
        mse_per_epoch = 0
        tic = time.time()
        # train stage
        lr = optimizer.param_groups[0]['lr']
        for ii, data in enumerate(data_loader):
            Xma, Xgt, XLI, mask = [x.cuda() for x in data]
            net.train()
            optimizer.zero_grad()
            X0, ListX, ListA= net(Xma, XLI, mask)
            loss_l2Xs = 0
            loss_l1Xs = 0
            loss_l1As = 0
            newAgt = mask*(Xma - Xgt)
            newXgt = mask * Xgt
            for j in range(opt.S):
                loss_l2Xs = loss_l2Xs +  0.1 * F.mse_loss(ListX[j]*mask, newXgt)
                loss_l1Xs = loss_l1Xs +  0.1 * torch.sum(torch.abs(ListX[j]*mask - newXgt))
                loss_l1As = loss_l1As +  0.1 * torch.sum(torch.abs(mask *ListA[j]-newAgt))
            loss_l1Xf = torch.sum(torch.abs((ListX[-1]*mask - newXgt)))
            loss_l1Af = torch.sum(torch.abs(mask *ListA[-1]-newAgt))
            loss_l2Xf = F.mse_loss(ListX[-1]*mask, newXgt)
            loss_l1X0=  0.1 * torch.sum(torch.abs(X0 * mask - newXgt))
            loss_l2X0 = 0.1 * F.mse_loss(X0 *mask, newXgt)
            loss_l2X = loss_l2Xs + loss_l2Xf + loss_l2X0
            loss_l1X = loss_l1Xs + loss_l1Xf + loss_l1X0
            loss_l1A =  loss_l1As + loss_l1Af
            loss = opt.Xl2 * loss_l2X + opt.Xl1* loss_l1X + opt.Al1 * loss_l1A
            # back propagation
            loss.backward()
            optimizer.step()
            mse_iter = loss.item()
            mse_per_epoch+= mse_iter
            if ii % 100 == 0:
                template = '[Epoch:{:>2d}/{:<2d}] {:0>5d}/{:0>5d}, Loss={:5.2e}, Lossl2X={:5.2e},  Lossl1X={:5.2e}, Lossl1A={:5.2e}, lr={:.2e}'
                print(template.format(epoch + 1, opt.niter, ii, num_iter_epoch, mse_iter, loss_l2X, loss_l1X, loss_l1A, lr))
            writer.add_scalar('Train Loss Iter', mse_iter, step)
            writer.add_scalar('lossl2X', loss_l2X.item(), step)
            writer.add_scalar('lossl1A', loss_l1A.item(), step)
            writer.add_scalar('lossl1X', loss_l1X.item(), step)
            writer.add_scalar('lossl2Xf', loss_l2Xf.item(), step)
            writer.add_scalar('lossl1Af', loss_l1Af.item(), step)
            writer.add_scalar('lossl1Xf', loss_l1Xf.item(), step)
            step += 1
        mse_per_epoch /= (ii + 1)
        print('Loss={:+.2e}'.format(mse_per_epoch))
        print('-' * 100)
        # adjust the learning rate
        lr_scheduler.step()
        # save model
        torch.save(net.state_dict(), os.path.join(opt.model_dir, 'DICDNet_latest.pt'))
        if epoch % 10 == 0:
            # save model
            model_prefix = 'model_'
            save_path_model = os.path.join(opt.model_dir, model_prefix + str(epoch + 1))
            torch.save({
                'epoch': epoch + 1,
                'step': step + 1,
            }, save_path_model)
            torch.save(net.state_dict(), os.path.join(opt.model_dir, 'DICDNet_%d.pt' % (epoch+1)))
        toc = time.time()
        print('This epoch take time {:.2f}'.format(toc - tic))
    writer.close()
    print('Reach the maximal epochs! Finish training')


if __name__ == '__main__':
    def print_network(net):
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
      #  print(net)
        print('Total number of parameters: %d' % num_params)
    net = DICDNet(opt).cuda()
    print_network(net)
    optimizer = optim.Adam(net.parameters(), betas=(0.5, 0.999), lr=opt.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestone,
                                                 gamma=0.5)  # learning rates
    # from opt.resume continue to train
    for _ in range(opt.resume):
        scheduler.step()
    if opt.resume:
        checkpoint = torch.load(os.path.join(opt.model_dir, 'model_' + str(opt.resume)))
        net.load_state_dict(torch.load(os.path.join(opt.model_dir, 'DICDNet_' + str(opt.resume) + '.pt')))
        print('loaded checkpoints, epoch{:d}'.format(checkpoint['epoch']))

    # load dataset
    train_mask = np.load(os.path.join(opt.data_path, 'trainmask.npy'))
    train_dataset = MARTrainDataset(opt.data_path, opt.patchSize, int(opt.batchSize * opt.batchnum), train_mask)
    # train model
    train_model(net, optimizer, scheduler, train_dataset)


