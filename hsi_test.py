import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from utils import *
from hsi_setup import Engine, train_options, make_dataset
import time

if __name__ == '__main__':
    """Training settings"""
    parser = argparse.ArgumentParser(
        description='Hyperspectral Image Denoising (Complex noise)')
    opt = train_options(parser)
    print(opt)
    

    """Setup Engine"""
    engine = Engine(opt)

    """Dataset Setting"""
    
    HSI2Tensor = partial(HSI2Tensor, use_2dconv=engine.get_net().use_2dconv)

   
    target_transform = HSI2Tensor()

    """Test-Dev"""

    test_dir = opt.test_dir#python hside_simu_test.py -a diffusion -p diffusion_mamba_gaussian -r -rp /home/wuzhehui/Hyper_CS/SERT-master/checkpoints/diffusion/diffusion_mamba_gaussian/model_epoch_90_298170.pth --test-dir /home/wuzhehui/Hyper_CS/denoising_train/data/ICVL/data_patch_db/Testing
    save_dir = opt.save_dir 
    sigmas = [10,30]

    if not engine.get_net().use_2dconv:
        test_transform =  Compose([
            AddNoiseNoniid(sigmas),
            SequentialSelect(
                transforms=[
                    lambda x:x,
                    #AddNoiseImpulse(),
                    #AddNoiseStripe(),
                    #AddNoiseDeadline(),
                    #AddNoiseComplex()
                ]
            ),
            HSI2Tensor()
        ])
    else:
        test_transform =  Compose([
            AddNoiseNoniid(sigmas),
            SequentialSelect(
                transforms=[
                    lambda x: x,
                    #AddNoiseImpulse(),
                    #AddNoiseStripe(),
                    #AddNoiseDeadline(),
                    #AddNoiseComplex()
                ]
            ),
            HSI2Tensor()
        ])

    target_transform = HSI2Tensor()

    mat_dataset = Dataset_test(data_dir = test_dir,suffix='.mat', test_transform = test_transform, target_transform = target_transform)
    mat_loader = DataLoader(mat_dataset,batch_size=1, shuffle=False,num_workers=1, pin_memory=opt.no_cuda)   

    strart_time = time.time()
    engine.test(mat_loader,save_dir)
    end_time = time.time()
    test_time = end_time-strart_time
    print('cost-time: ',(test_time/len(mat_dataset)))
