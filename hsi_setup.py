from email.mime import base, image
from locale import normalize
from math import fabs
from xml.sax import SAXException
import torch
import torch.optim as optim
import models

import os
import argparse
           
from os.path import join
from utils import *
from thop import profile
from torchstat import stat 
import scipy.io as scio
from models.ssdm import diffusion

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torchvision import models as torchmodel
from torchvision import  utils
from torch import  einsum

import torchvision.utils as vutil
from pathlib import Path
import time

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

class MultipleLoss(nn.Module):
    def __init__(self, losses, weight=None):
        super(MultipleLoss, self).__init__()
        self.losses = nn.ModuleList(losses)
        self.weight = weight or [1/len(self.losses)] * len(self.losses)
    
    def forward(self, predict, target):
        total_loss = 0
        for weight, loss in zip(self.weight, self.losses):
            total_loss += loss(predict, target) * weight
        return total_loss

    def extra_repr(self):
        return 'weight={}'.format(self.weight)

def train_options(parser):
    def _parse_str_args(args):
        str_args = args.split(',')
        parsed_args = []
        for str_arg in str_args:
            arg = int(str_arg)
            if arg >= 0:
                parsed_args.append(arg)
        return parsed_args    
    parser.add_argument('--prefix', '-p', type=str, default='denoise',
                        help='prefix')
    parser.add_argument('--arch', '-a', metavar='ARCH', required=True,
                        choices=                                                         model_names        ,
                        help='model architecture: ' +
                        ' | '.join(model_names))
    parser.add_argument('--batchSize', '-b', type=int,
                        default=16, help='training batch size. default=16')         
    parser.add_argument('--lr', type=float, default=1e-6,
                        help='learning rate. default=1e-3.')
    parser.add_argument('--wd', type=float, default=0,
                        help='weight decay. default=0')
    parser.add_argument('--loss', type=str, default='l2',
                        help='which loss to choose.', choices=['l1', 'l2', 'smooth_l1', 'ssim', 'l2_ssim'])
    parser.add_argument('--testdir', type=str)
    parser.add_argument('--sigma', type=int)
    parser.add_argument('--init', type=str, default='xu',
                        help='which init scheme to choose.', choices=['kn', 'ku', 'xn', 'xu', 'edsr'])
    parser.add_argument('--no-cuda', action='store_true', help='disable cuda?')
    parser.add_argument('--no-log', action='store_true',
                        help='disable logger?')
    parser.add_argument('--threads', type=int, default=1,
                        help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=2024,
                        help='random seed to use. default=2024')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--no-ropt', '-nro', action='store_true',
                            help='not resume optimizer')          
    parser.add_argument('--chop', action='store_true',
                            help='forward chop')                                      
    parser.add_argument('--resumePath', '-rp', type=str,
                        default=None, help='checkpoint to use.')
    parser.add_argument('--dataroot', '-d', type=str,
                        default='/data/HSI_Data/ICVL64_31.db', help='data root')
    parser.add_argument('--train-dir', type=str,
                    default='', help='The path of train dataset')
    parser.add_argument('--valid-dir', type=str,
                    default='', help='The path of valid dataset')
    parser.add_argument('--test-dir', type=str,
                    default='', help='The path of test dataset')
    parser.add_argument('--save-dir', type=str,
                    default='', help='The path of save address')
    parser.add_argument('--save', action='store_true', help='Whether to save intermediate results')
    parser.add_argument('--clip', type=float, default=1e6)
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids')
    parser.add_argument('--time_steps', type=int, default=1000, help='time steps used for diffusion')
    parser.add_argument('--sample_steps', type=int, default=1, help='time steps used for sample')
    parser.add_argument('--acc_steps', type=int, default=1, help='accumulation_steps')
    parser.add_argument('--image_size', type=int, default=64, help='image size')
    parser.add_argument('--channels', type=int, default=31, help='number of bands')
    

    opt = parser.parse_args()
    opt.gpu_ids = _parse_str_args(opt.gpu_ids)

    return opt


def make_dataset(opt, train_transform, target_transform, common_transform, batch_size=None, repeat=1):
    dataset = LMDBDataset(opt.dataroot, repeat=repeat)
    # dataset.length -= 1000
    # dataset.length = size or dataset.length

    """Split patches dataset into training, validation parts"""
    dataset = TransformDataset(dataset, common_transform)

    train_dataset = ImageTransformDataset(dataset, train_transform, target_transform)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size or opt.batchSize, shuffle=True,
                              num_workers=opt.threads, pin_memory=not opt.no_cuda, worker_init_fn=worker_init_fn)

    return train_loader

class Engine(object):
    def __init__(self, opt):
        self.prefix = opt.prefix
        self.opt = opt
        self.net = None
        self.optimizer = None
        self.criterion = None
        self.basedir = None
        self.iteration = None
        self.epoch = None
        self.best_psnr = None
        self.best_loss = None
        self.writer = None
        self.devide_ids = opt.gpu_ids

        self.__setup()

    def __setup(self):


        self.basedir = join('checkpoints', self.opt.arch)
        if not os.path.exists(self.basedir):
            os.makedirs(self.basedir)

        self.best_psnr = 0
        self.best_loss = 1e6
        self.epoch = 0  # start from epoch 0 or last checkpoint epoch
        self.iteration = 0

        cuda = not self.opt.no_cuda
        self.device = torch.device("cuda") if cuda else 'cpu'

        print('Cuda Acess: %d' % cuda)
        if cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")

        torch.manual_seed(self.opt.seed)
        if cuda:
            torch.cuda.manual_seed(self.opt.seed)

        """Model"""
        print("=> creating model '{}'".format(self.opt.arch))
        self.net = models.__dict__[self.opt.arch]()
        self.diffusion = None
        # initialize parameters

        init_params(self.net, init_type=self.opt.init) # disable for default initialization
        
        if self.opt.loss == 'l2':
            self.criterion = nn.MSELoss()
        if self.opt.loss == 'l1':
            self.criterion = nn.L1Loss()
        if self.opt.loss == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss()
        if self.opt.loss == 'ssim':
            self.criterion = SSIMLoss(data_range=1, channel=31)
        if self.opt.loss == 'l2_ssim':
            self.criterion = MultipleLoss([nn.MSELoss(), SSIMLoss(data_range=1, channel=31)], weight=[1, 2.5e-3])

        print(self.criterion)

        # if len(self.opt.gpu_ids) > 1:
        #     self.net  = nn.DataParallel(self.net.cuda(), device_ids=self.opt.gpu_ids, output_device=self.opt.gpu_ids[0])
        # if cuda:
        #     self.net.to(self.device)
        #     print('cuda initialized')
        #     self.criterion = self.criterion.to(self.device)
        
        if cuda:
            if len(self.opt.gpu_ids) > 1:
                self.net  = nn.DataParallel(self.net.cuda(), device_ids=self.opt.gpu_ids)
            self.net = self.net.cuda()
            print('cuda initialized')
            self.criterion = self.criterion.cuda()
        """Logger Setup"""
        log = not self.opt.no_log
        if log:
            self.writer = get_summary_writer(os.path.join(self.basedir, 'logs'), self.opt.prefix)

        """Optimization Setup"""
        # self.optimizer = optim.AdamW(
        #     self.net.parameters(), lr=self.opt.lr, weight_decay=self.opt.wd, amsgrad=False)
        self.optimizer = optim.Adam(
            self.net.parameters(), lr=self.opt.lr, weight_decay=self.opt.wd, amsgrad=False)
        """Resume previous model"""
        if self.opt.resume:
            # Load checkpoint.
            self.load(self.opt.resumePath, not self.opt.no_ropt)
        else:
            print('==> Building model..')
            print(self.net)

    def forward(self, inputs):        
        if self.opt.chop:            
            output = self.forward_chop(inputs)
        else:
            output = self.net(inputs)
        
        return output

    def forward_chop(self, x, base=16):        
        n, c, b, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        
        shave_h = np.ceil(h_half / base) * base - h_half
        shave_w = np.ceil(w_half / base) * base - w_half

        shave_h = shave_h if shave_h >= 10 else shave_h + base
        shave_w = shave_w if shave_w >= 10 else shave_w + base

        h_size, w_size = int(h_half + shave_h), int(w_half + shave_w)        
        
        inputs = [
            x[..., 0:h_size, 0:w_size],
            x[..., 0:h_size, (w - w_size):w],
            x[..., (h - h_size):h, 0:w_size],
            x[..., (h - h_size):h, (w - w_size):w]
        ]

        outputs = [self.net(input_i) for input_i in inputs]

        output = torch.zeros_like(x)
        output_w = torch.zeros_like(x)
        
        output[..., 0:h_half, 0:w_half] += outputs[0][..., 0:h_half, 0:w_half]
        output_w[..., 0:h_half, 0:w_half] += 1
        output[..., 0:h_half, w_half:w] += outputs[1][..., 0:h_half, (w_size - w + w_half):w_size]
        output_w[..., 0:h_half, w_half:w] += 1
        output[..., h_half:h, 0:w_half] += outputs[2][..., (h_size - h + h_half):h_size, 0:w_half]
        output_w[..., h_half:h, 0:w_half] += 1
        output[..., h_half:h, w_half:w] += outputs[3][..., (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]
        output_w[..., h_half:h, w_half:w] += 1
        
        output /= output_w

        return output

    def __step_diffusion(self, train, batch_idx, acc_steps, inputs, targets,sigma=None):        
        loss_data = 0
        total_norm = None 
        loss = torch.mean(self.diffusion(targets, inputs)) 
        loss = loss/acc_steps
        if train:
            loss.backward()#
            total_norm = nn.utils.clip_grad_norm_(self.net.parameters(), self.opt.clip)
            if (batch_idx+1) % acc_steps == 0:
                self.optimizer.step()#
                self.optimizer.zero_grad()
        loss_data += loss.item()

        return loss_data, total_norm

    def  __step(self, train, batch_idx, acc_steps, inputs, targets,sigma=None):        
        if train:
            self.optimizer.zero_grad()#
        loss = 0
        loss_data = 0
        total_norm = None
        # self.net.eval()
        if self.get_net().bandwise:
            O = []
            for time, (i, t) in enumerate(zip(inputs.split(1, 1), targets.split(1, 1))):
                o = self.net(i)
                O.append(o)
                loss = self.criterion(o, t)
            outputs = torch.cat(O, dim=1)
        else:
           
            #noisy_sigma = torch.zeros
            if self.opt.arch == 'ssdm' or self.opt.arch == 'ssdm_urban':
                outputs = self.diffusion.module.sample(batches=self.opt.batchSize, img_size = 64,condition_img = inputs)
                loss = torch.mean(self.diffusion(targets, inputs)) 
            elif self.opt.arch == 'hsid':
                inputs1 = inputs[:,0,:,:].unsqueeze(1)
                inputs2 = inputs[:,1:31,:,:]
                outputs = self.net(inputs1,inputs2)
                loss += self.criterion(outputs, targets)
            else:
                outputs = self.net(inputs)
                #outputs = inputs

                loss = self.criterion(outputs[...], targets) 

        if train:
            loss.backward()
            total_norm = nn.utils.clip_grad_norm_(self.net.parameters(), self.opt.clip)
            self.optimizer.step()#
            if (batch_idx+1) % acc_steps == 0:
                self.optimizer.step()#
                self.optimizer.zero_grad()            
        loss_data += loss.item()

        return outputs, loss_data, total_norm

    def load(self, resumePath=None, load_opt=True):
        print('==> Resuming from checkpoint %s..' % resumePath)
        assert os.path.isdir('checkpoints'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(resumePath, map_location=torch.device('cuda:0'))
        if load_opt:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epoch = checkpoint['epoch']
        self.iteration = checkpoint['iteration']
        self.get_net().load_state_dict(checkpoint['net'])

        

    def train(self, train_loader):
        print('\nEpoch: %d' % self.epoch)
        self.net.train()

        if self.opt.arch == 'ssdm' or self.opt.arch == 'ssdm_urban' or self.opt.arch == 'unet':
            self.diffusion = diffusion.GaussianDiffusion(
                self.net,
                image_size = self.opt.image_size,
                timesteps = self.opt.time_steps,   
                loss_type = self.opt.loss,    
                sampling_timesteps = self.opt.sample_steps,
                objective = 'pred_x0',
                beta_schedule = 'linear',
                p2_loss_weight_gamma = 1,
                p2_loss_weight_k = 1,
                ddim_sampling_eta = 1.
            ).cuda()

        train_loss = 0
        train_psnr = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if not self.opt.no_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            if self.opt.arch == 'ssdm' or self.opt.arch == 'ssdm_urban' or self.opt.arch == 'unet':
                loss_data, total_norm = self.__step_diffusion(True, batch_idx, self.opt.acc_steps, inputs, targets)
                train_loss += loss_data
                avg_loss = train_loss / (batch_idx+1)
                if not self.opt.no_log:
                    self.writer.add_scalar(
                        join(self.prefix, 'train_loss'), loss_data, self.iteration)
                    self.writer.add_scalar(
                        join(self.prefix, 'train_avg_loss'), avg_loss, self.iteration)
                progress_bar(batch_idx, len(train_loader), 'AvgLoss: %.4e | Loss: %.4e | Norm: %.4e' 
                        % (avg_loss, loss_data, total_norm))
            elif self.opt.arch == 'hsid':
                rand = inputs.shape[-3]
                for k in range(rand):
                    inputs1 = inputs[:,k,:,:].unsqueeze(1)#16,1,64,64
                    target = targets[:,k,:,:].unsqueeze(1)#16,1,64,64
                    if k < 12:
                        inputs2 = inputs[:,0:24, :, :]
                    elif k > 18:
                        inputs2 = inputs[:,6:30, :, :]
                    # if k < 80:
                    #     inputs2 = inputs[:,0:120, :, :]
                    # elif k > 120:
                    #     inputs2 = inputs[:,89:209, :, :]   
                    # else:
                    #     inputs2 = inputs[:,left:right, :, :]                 
                    input = torch.cat((inputs1,inputs2),dim = 1)
                    _, loss_data, total_norm = self.__step(True, batch_idx, self.opt.acc_steps, input, target)
                    #print(inputs1.shape,inputs2.shape,input.shape,target.shape)
                train_loss += loss_data
                avg_loss = train_loss / (batch_idx+1)
                if not self.opt.no_log:
                    self.writer.add_scalar(
                        join(self.prefix, 'train_loss'), loss_data, self.iteration)
                    self.writer.add_scalar(
                        join(self.prefix, 'train_avg_loss'), avg_loss, self.iteration)
                progress_bar(batch_idx, len(train_loader), 'AvgLoss: %.4e | Loss: %.4e | Norm: %.4e' 
                        % (avg_loss, loss_data, total_norm))
            else:
                outputs, loss_data, total_norm = self.__step(True, batch_idx, self.opt.acc_steps, inputs, targets)
                train_loss += loss_data
                avg_loss = train_loss / (batch_idx+1)
                if not self.opt.no_log:
                    self.writer.add_scalar(
                        join(self.prefix, 'train_loss'), loss_data, self.iteration)
                    self.writer.add_scalar(
                        join(self.prefix, 'train_avg_loss'), avg_loss, self.iteration)
                progress_bar(batch_idx, len(train_loader), 'AvgLoss: %.4e | Loss: %.4e | Norm: %.4e' 
                        % (avg_loss, loss_data, total_norm))

            self.iteration += 1

        self.epoch += 1
        if not self.opt.no_log:
            self.writer.add_scalar(
                join(self.prefix, 'train_loss_epoch'), avg_loss, self.epoch)

  
    def test(self, test_loader,save_dir):
        self.net.eval()

        if self.opt.arch == 'ssdm' or self.opt.arch == 'ssdm_urban' or self.opt.arch == 'unet':
            self.diffusion = diffusion.GaussianDiffusion(
                self.net,
                image_size = self.opt.image_size,
                channels = self.opt.channels,
                timesteps = self.opt.time_steps,   
                loss_type = self.opt.loss,    
                sampling_timesteps = self.opt.sample_steps,
                objective = 'pred_x0',
                beta_schedule = 'linear',
                p2_loss_weight_gamma = 1,
                p2_loss_weight_k = 1,
                ddim_sampling_eta = 1.
            ).cuda()

        test_loss = 0
        total_psnr = 0
        total_ssim = 0
        total_sam = 0
        total_lpips = 0
        # total_niqe = 0

        print(len(test_loader))
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                if not self.opt.no_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                    inputs_np = inputs.cpu().numpy()
                    print(inputs_np.shape)
                    # 保存为 .mat 文件
                    #sio.savemat(f'/home/wuzhehui/Hyper_Denoising/SSDM-main-copy/complex_degrad/complex/degrad.mat', {'degrad': inputs_np[0]})
                    #sio.savemat(f'/home/wuzhehui/Hyper_Denoising/SSDM-main-copy/urban/degrad.mat', {'degrad': inputs_np[0]})
                if self.opt.arch == 'ssdm' or self.opt.arch == 'ssdm_urban' or self.opt.arch == 'unet':
                    outputs = self.diffusion.sample(batches=1, img_size = self.opt.image_size,condition_img = inputs, save = self.opt.save, save_dir = save_dir)
                    loss_data = self.criterion(targets,outputs).item()
                elif self.opt.arch == 'hsid':
                    outputs = torch.randn_like(inputs)
                    rand = inputs.shape[-3]
                    for k in range(rand):
                        inputs1 = inputs[:,k,:,:].unsqueeze(1)#16,1,64,64
                        target = targets[:,k,:,:].unsqueeze(1)#16,1,64,64
                        # if k < 80:
                        #     inputs2 = inputs[:,0:120, :, :]
                        # elif k > 120:
                        #     inputs2 = inputs[:,89:209, :, :]   
                        if k < 12:
                            inputs2 = inputs[:,0:24, :, :]
                        elif k > 18:
                            inputs2 = inputs[:,6:30, :, :]   
                        input = torch.cat((inputs1,inputs2),dim = 1)
                        output, loss_data, total_norm = self.__step(False, batch_idx, self.opt.acc_steps, input, target)
                        outputs[:,k,:,:] = output
                else:
                    outputs, loss_data, _ = self.__step(False, batch_idx, self.opt.acc_steps, inputs, targets)
                
                h,w=inputs.shape[-2:]
                band = inputs.shape[-3]
                result = outputs.squeeze().cpu().detach().numpy()
                img = targets.squeeze().cpu().detach().numpy()
                sio.savemat(f'/home/wuzhehui/Hyper_Denoising/SSDM-main-copy/output/gt/{batch_idx}.mat', {'restored': result})
                #sio.savemat(f'/home/wuzhehui/Hyper_Denoising/SSDM-main-copy/urban/sst_30.mat', {'restored': result})
                # if self.opt.channels == 31:
                #     lpips = []
                #     if not self.net.use_2dconv:  
                #         for k in range(band//3):
                #             lpips.append(cal_lpips(outputs[0,0,[k+20,k+10,k],:,:].unsqueeze(0),targets[0,0,[k+20,k+10,k],:,:].unsqueeze(0)))
                #             LPIPS = sum(lpips)/len(lpips)
                #             #niqe = calculate_niqe(outputs[0,0,[5,10,30],:,:].cpu().numpy().transpose(1,2,0)*255, 0, input_order='HWC', convert_to='y').squeeze()
                #     else:
                #         for k in range(band//3):
                #             lpips.append(cal_lpips(outputs[0,[k+20,k+10,k],:,:].unsqueeze(0),targets[0,[k+20,k+10,k],:,:].unsqueeze(0)))
                #             LPIPS = sum(lpips)/len(lpips)
                #             #niqe = calculate_niqe(outputs[0,[5,10,30],:,:].cpu().numpy().transpose(1,2,0)*255, 0, input_order='HWC', convert_to='y').squeeze()                    
                # else:    
                #     LPIPS = 0
                LPIPS = 0

                psnr = [] 
                for k in range(band):
                    psnr.append(10*np.log10((h*w)/sum(sum((result[k]-img[k])**2))))
                PSNR = sum(psnr)/len(psnr)

                ssim = []
                k1 = 0.01
                k2 = 0.03
                for k in range(band):
                    ssim.append((2*np.mean(result[k])*np.mean(img[k])+k1**2) \
                        *(2*np.cov(result[k].reshape(h*w), img[k].reshape(h*w))[0,1]+k2**2) \
                        /(np.mean(result[k])**2+np.mean(img[k])**2+k1**2) \
                        /(np.var(result[k])+np.var(img[k])+k2**2))
                SSIM = (sum(ssim)/len(ssim))

                temp = (np.sum(result*img, 0) + np.spacing(1)) \
                    /(np.sqrt(np.sum(result**2, 0) + np.spacing(1))) \
                    /(np.sqrt(np.sum(img**2, 0) + np.spacing(1)))

                sam = np.mean(np.arccos(temp))*180/np.pi
                SAM = sam

                test_loss += loss_data
                total_psnr += PSNR
                total_ssim += SSIM
                total_sam += SAM
                total_lpips += LPIPS
                #total_niqe += niqe
                avg_loss = test_loss / (batch_idx+1)
                avg_psnr = total_psnr / (batch_idx+1)
                avg_ssim = total_ssim / (batch_idx+1)
                avg_sam = total_sam / (batch_idx+1)
                avg_lpips = total_lpips / (batch_idx+1)
                #avg_niqe = total_niqe / (batch_idx+1)

                progress_bar(batch_idx, len(test_loader), 'Loss: %.4e | PSNR: %.4f | \n SSIM: %.4f |SAM: %.4f | LPIPS: %.4f \n'
                          % (avg_loss, PSNR, SSIM, SAM, LPIPS))
        
        print(avg_loss, avg_psnr, avg_ssim, avg_sam, avg_lpips)
        return avg_psnr, avg_loss, avg_sam, avg_lpips

    def validate(self, valid_loader):
        self.net.eval()

        if self.opt.arch == 'ssdm' or self.opt.arch == 'ssdm_urban':
            self.diffusion = diffusion.GaussianDiffusion(
                self.net,
                image_size = self.opt.image_size,
                timesteps = self.opt.time_steps,   
                loss_type = self.opt.loss,    
                sampling_timesteps = self.opt.sample_steps,
                objective = 'pred_x0',
                beta_schedule = 'linear',
                p2_loss_weight_gamma = 1,
                p2_loss_weight_k = 1,
                ddim_sampling_eta = 1.
            ).cuda()

        test_loss = 0
        total_psnr = 0
        total_ssim = 0
        total_sam = 0
        total_lpips = 0
        # total_niqe = 0

        print(len(valid_loader))
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valid_loader):
                if not self.opt.no_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()

                if self.opt.arch == 'ssdm' or self.opt.arch == 'ssdm_urban':
                    outputs = self.diffusion.sample(batches=1, img_size = self.opt.image_size,condition_img = inputs,targets = targets)
                    loss_data = self.criterion(targets,outputs).item()
                elif self.opt.arch == 'hsid':
                    outputs = torch.randn_like(inputs)
                    rand = inputs.shape[-3]
                    for k in range(rand):
                        inputs1 = inputs[:,k,:,:].unsqueeze(1)#16,1,64,64
                        target = targets[:,k,:,:].unsqueeze(1)#16,1,64,64
                        # if k < 80:
                        #     inputs2 = inputs[:,0:120, :, :]
                        # elif k > 120:
                        #     inputs2 = inputs[:,89:209, :, :]   
                        if k < 12:
                            inputs2 = inputs[:,0:24, :, :]
                        elif k > 18:
                            inputs2 = inputs[:,6:30, :, :]   
                        input = torch.cat((inputs1,inputs2),dim = 1)
                        output, loss_data, total_norm = self.__step(False, batch_idx, self.opt.acc_steps, input, target)
                        outputs[:,k,:,:] = output
                else:
                    outputs, loss_data, _ = self.__step(False, batch_idx, self.opt.acc_steps, inputs, targets)
           
                h,w=inputs.shape[-2:]
                band = inputs.shape[-3]
                result = outputs.squeeze().cpu().detach().numpy()
                img = targets.squeeze().cpu().detach().numpy()

                lpips = []
                if not self.net.use_2dconv:  
                    for k in range(band//3):
                        lpips.append(cal_lpips(outputs[0,0,[k+20,k+10,k],:,:].unsqueeze(0),targets[0,0,[k+20,k+10,k],:,:].unsqueeze(0)))
                        LPIPS = sum(lpips)/len(lpips)
                        #niqe = calculate_niqe(outputs[0,0,[5,10,30],:,:].cpu().numpy().transpose(1,2,0)*255, 0, input_order='HWC', convert_to='y').squeeze()
                else:
                    for k in range(band//3):
                        lpips.append(cal_lpips(outputs[0,[k+20,k+10,k],:,:].unsqueeze(0),targets[0,[k+20,k+10,k],:,:].unsqueeze(0)))
                        LPIPS = sum(lpips)/len(lpips)
                        #niqe = calculate_niqe(outputs[0,[5,10,30],:,:].cpu().numpy().transpose(1,2,0)*255, 0, input_order='HWC', convert_to='y').squeeze()                    
    
                psnr = [] 
                for k in range(band):
                    psnr.append(10*np.log10((h*w)/sum(sum((result[k]-img[k])**2))))
                PSNR = sum(psnr)/len(psnr)

                ssim = []
                k1 = 0.01
                k2 = 0.03
                for k in range(band):
                    ssim.append((2*np.mean(result[k])*np.mean(img[k])+k1**2) \
                        *(2*np.cov(result[k].reshape(h*w), img[k].reshape(h*w))[0,1]+k2**2) \
                        /(np.mean(result[k])**2+np.mean(img[k])**2+k1**2) \
                        /(np.var(result[k])+np.var(img[k])+k2**2))
                SSIM = (sum(ssim)/len(ssim))

                temp = (np.sum(result*img, 0) + np.spacing(1)) \
                    /(np.sqrt(np.sum(result**2, 0) + np.spacing(1))) \
                    /(np.sqrt(np.sum(img**2, 0) + np.spacing(1)))

                sam = np.mean(np.arccos(temp))*180/np.pi
                SAM = sam

                test_loss += loss_data
                total_psnr += PSNR
                total_ssim += SSIM
                total_sam += SAM
                total_lpips += LPIPS
                #total_niqe += niqe
                avg_loss = test_loss / (batch_idx+1)
                avg_psnr = total_psnr / (batch_idx+1)
                avg_ssim = total_ssim / (batch_idx+1)
                avg_sam = total_sam / (batch_idx+1)
                avg_lpips = total_lpips / (batch_idx+1)
                #avg_niqe = total_niqe / (batch_idx+1)

                progress_bar(batch_idx, len(valid_loader), 'Loss: %.4e | PSNR: %.4f | \n SSIM: %.4f |SAM: %.4f | LPIPS: %.4f \n'
                          % (avg_loss, PSNR, SSIM, SAM, LPIPS))
        
        print(avg_loss, avg_psnr, avg_ssim, avg_sam, avg_lpips)
        return avg_psnr, avg_loss, avg_sam, avg_lpips

  
    def save_checkpoint(self, model_out_path=None, **kwargs):
        if not model_out_path:
            model_out_path = join(self.basedir, self.prefix, "model_epoch_%d_%d.pth" % (
                self.epoch, self.iteration))

        state = {
            'net': self.get_net().state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'iteration': self.iteration,
        }
        
        state.update(kwargs)

        if not os.path.isdir(join(self.basedir, self.prefix)):
            os.makedirs(join(self.basedir, self.prefix))

        torch.save(state, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    # saving result into disk
    def test_develop(self, test_loader, savedir=None, verbose=True):
        from scipy.io import savemat
        from os.path import basename, exists

        def torch2numpy(hsi):
            if self.net.use_2dconv:
                R_hsi = hsi.data[0].cpu().numpy().transpose((1,2,0))
            else:
                R_hsi = hsi.data[0].cpu().numpy()[0,...].transpose((1,2,0))
            return R_hsi    

        self.net.eval()
        test_loss = 0

    def get_net(self):
        if len(self.opt.gpu_ids) > 1:
            return self.net.module
        else:
            return self.net 
  