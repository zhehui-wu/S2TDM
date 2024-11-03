import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
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
    train_transform = Compose([
        AddNoiseNoniidv2(0,55),
        HSI2Tensor()
    ])

    train_dir  = opt.train_dir#'/home/wuzhehui/Hyper_CS/denoising_train/data/APEX/data_patch_db/Training/APEX64_210.db'
    train_data = LMDBDataset(train_dir,repeat=10)
    
    target_transform = HSI2Tensor()
    train_dataset = ImageTransformDataset(train_data, train_transform,target_transform)


    train_loader = DataLoader(train_dataset,
                              batch_size=opt.batchSize, shuffle=True,
                              num_workers=opt.threads, pin_memory=not opt.no_cuda, worker_init_fn=worker_init_fn)
    
    base_lr = opt.lr
    epoch_per_save = 5
    adjust_learning_rate(engine.optimizer, opt.lr)

    engine.epoch  = 0
    while engine.epoch < 100:
        np.random.seed()
        
        if engine.epoch == 100:
            adjust_learning_rate(engine.optimizer, base_lr*0.1)

        engine.train(train_loader)


        display_learning_rate(engine.optimizer)
        print('Latest Result Saving...')
        model_latest_path = os.path.join(engine.basedir, engine.prefix, 'model_latest.pth')
        engine.save_checkpoint(
            model_out_path=model_latest_path
        )

        display_learning_rate(engine.optimizer)
        if engine.epoch % epoch_per_save == 0:
            engine.save_checkpoint()
