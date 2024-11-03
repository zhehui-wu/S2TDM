import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import argparse
from utils import *
import datetime
from hsi_setup import Engine, train_options
# import wandb

if __name__ == '__main__':

    """Training settings"""
    #torch.set_num_threads(2)
    parser = argparse.ArgumentParser(
    description='Hyperspectral Image Denoising (Gaussian noise)')
    opt = train_options(parser)#参数设置
    print(opt)

    data = datetime.datetime.now()

    """Setup Engine"""
    engine = Engine(opt)#Trainer

    """Dataset Setting"""
    
    HSI2Tensor = partial(HSI2Tensor, use_2dconv=engine.get_net().use_2dconv)

    target_transform = HSI2Tensor()

    train_transform = Compose([
        AddNoiseBlindv2(10,70),
        HSI2Tensor()
    ])

    train_dir = opt.train_dir

    train_data = LMDBDataset(train_dir) 
    
    target_transform = HSI2Tensor()
    train_dataset = ImageTransformDataset(train_data, train_transform,target_transform)

    train_loader = DataLoader(train_dataset,
                              batch_size=opt.batchSize, shuffle=True,
                              num_workers=opt.threads, pin_memory=not opt.no_cuda, worker_init_fn=worker_init_fn)

    valid_dir = opt.valid_dir#验证集

    target_transform = HSI2Tensor()       
    valid_transform = Compose([
        AddNoise(50),
        HSI2Tensor()
    ])

    mat_dataset = Dataset_test(data_dir = valid_dir,suffix='.mat', test_transform = valid_transform, target_transform = target_transform)
    mat_loader = DataLoader(mat_dataset,batch_size=1, shuffle=False,num_workers=1, pin_memory=opt.no_cuda)

    base_lr = opt.lr
    epoch_per_save = 5
    adjust_learning_rate(engine.optimizer, opt.lr)

    engine.epoch  = 0 
    while engine.epoch < 100:
        np.random.seed()

        if engine.epoch == 60:
            adjust_learning_rate(engine.optimizer, base_lr*0.1)

        engine.train(train_loader)
        engine.validate(mat_loader) 

        display_learning_rate(engine.optimizer)
        print('Latest Result Saving...')
        model_latest_path = os.path.join(engine.basedir, engine.prefix, 'model_latest.pth')
        engine.save_checkpoint(
            model_out_path=model_latest_path
        )

        display_learning_rate(engine.optimizer)
        if engine.epoch % epoch_per_save == 0:
            engine.save_checkpoint()

 
