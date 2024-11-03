import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import argparse
from utils import *
from hsi_setup import Engine, train_options


if __name__ == '__main__':
    """Training settings"""
    parser = argparse.ArgumentParser(
        description='Hyperspectral Image Denoising (Complex noise)')
    opt = train_options(parser)
    print(opt)
    

    """Setup Engine"""
    engine = Engine(opt)

    """Dataset Setting"""
    
    HSI2Tensor = partial(HSI2Tensor, use_2dconv=engine.net.use_2dconv)

   
    target_transform = HSI2Tensor()

    test_dir = opt.test_dir
    save_dir = opt.save_dir   

    test_transform = HSI2Tensor() 
    target_transform = HSI2Tensor()
    mat_dataset = Dataset_test(data_dir = test_dir,suffix='.mat', test_transform = test_transform, target_transform = target_transform)
    mat_loader = DataLoader(mat_dataset,batch_size=1, shuffle=False,num_workers=1, pin_memory=opt.no_cuda)    

    
    engine.test(mat_loader,save_dir)
        
