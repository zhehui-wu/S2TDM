"""generate testing mat dataset"""
import os
import numpy as np
import h5py
from os.path import join, exists
from scipy.io import loadmat, savemat
import scipy.io as sio
from tqdm import tqdm

from util import crop_center, Visualize3D, minmax_normalize, rand_crop
from PIL import Image
from spectral import *
from pathlib import Path

def crop_center(img,cropx,cropy):
    _,y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[:, starty:starty+cropy,startx:startx+cropx]

def create_mat_dataset(datadir, fnames, newdir, matkey, func=None, load=h5py.File):
    if not exists(newdir):
        os.mkdir(newdir)
    

    for i, fn in enumerate(fnames):
        print('generate data(%d/%d)' %(i+1, len(fnames)))
        filepath = join(datadir, fn)
        try:
            mat = load(filepath)
            
            data = func(mat[matkey][...])
            data_hwc = data.transpose((2,1,0))
            savemat(join(newdir, fn), {'data': data_hwc})
            # try:
            #     Image.fromarray(np.array(data_hwc*255,np.uint8)[:,:,20]).save('/home/wuzhehui/Hyper_CS/denoising_train/data/ICVL/data_patch_db/Testing/icvl_test_512_png/{}.png'.format(os.path.splitext(fn)[0]))
            # except Exception as e:
            #     print(e)
        except:
            print('open error for {}'.format(fn))
            continue

def create_ICVL_valid():
    save_path = Path('/home/wuzhehui/Hyper_CS/SSDM-main/data/dataset/ICVL/Validating')
    src_img_root_dir_path = Path("/home/wuzhehui/Hyper_CS/SSDM-main/data/ICVL/Validating")
    src_img_path_list = sorted(list(src_img_root_dir_path.glob('*')))
    crop_sizes=(512, 512)

    idx = 1
    for src_img_path in tqdm(src_img_path_list):
        data = h5py.File(src_img_path)['rad']
        new_data = []
        amin = np.min(data)
        amax = np.max(data)
        data = (data - amin) / (amax - amin)
        data = np.rot90(data, k=2, axes=(1,2)) # ICVL
        data = crop_center(data, crop_sizes[0], crop_sizes[1])
        img = data.astype(np.float32)#[0,1]
        print(idx,':',data.shape)
        sio.savemat(save_path / f'{idx}', {'img':img})
        idx+=1    

def create_ICVL_test():
    save_path = Path('/home/wuzhehui/Hyper_CS/SSDM-main/data/dataset/ICVL/Testing')
    src_img_root_dir_path = Path("/home/wuzhehui/Hyper_CS/SSDM-main/data/ICVL/Testing")
    src_img_path_list = sorted(list(src_img_root_dir_path.glob('*')))
    crop_sizes=(512, 512)

    idx = 1
    for src_img_path in tqdm(src_img_path_list):
        data = h5py.File(src_img_path)['rad']
        new_data = []
        amin = np.min(data)
        amax = np.max(data)
        data = (data - amin) / (amax - amin)
        data = np.rot90(data, k=2, axes=(1,2)) # ICVL
        data = crop_center(data, crop_sizes[0], crop_sizes[1])
        img = data.astype(np.float32)#[0,1]
        print(idx,':',data.shape)
        sio.savemat(save_path / f'{idx}', {'img':img})
        idx+=1   

def Preprocess_apex_dataset():
    imgpath = '/home/wuzhehui/Hyper_CS/SSDM-main/data/APEX/APEX_OSD_Package_1.0/APEX_OSD_V1_calibr_cube'
    
    img = open_image(imgpath+'.hdr')
    img = img.load()
    img = img.transpose((2,0,1))
    data = img[:210]
    total_num = 20
    print('processing---')
    save_dir = '/home/wuzhehui/Hyper_CS/SSDM-main/data/dataset/APEX/Training/'
    for i in range(total_num):
        data = rand_crop(data, 512, 512)
        data = minmax_normalize(data)
        data = data.transpose(1,2,0)*255
        data= truncated_linear_stretch(data,truncated_percent = 2).transpose(2,0,1)/255
        print(data.max())
        savemat(save_dir+str(i)+'.mat',{'data': data})
        print(i)

def create_Urban_test():
    imgpath = '/home/wuzhehui/Hyper_CS/SSDM-main/data/Urban/Urban_F210.mat'
    img = loadmat(imgpath)
    imgg  = img['Y'].reshape((210,307,307))
    imggt = imgg.astype(np.float32)
    norm_gt = imggt.transpose((1,2,0))
    norm_gt = norm_gt[:304,:304,:]
    norm_gt = minmax_normalize(norm_gt)*255
    norm_gt= truncated_linear_stretch(norm_gt*255,truncated_percent = 2).transpose(2,0,1)/255
    print(norm_gt.max())
    savemat("/home/wuzhehui/Hyper_CS/SSDM-main/data/dataset/Urban/Testing/Urban_F210.mat", {'gt': norm_gt})

def truncated_linear_stretch(
    image, truncated_percent=2, stretch_range=[0, 255], is_drop_non_positive=False
):
    """_summary_

    Args:
        image (np.array): HWC or HW
        truncated_percent (int, optional): _description_. Defaults to 2.
        stretch_range (list, optional): _description_. Defaults to [0, 255].
    """
    max_tansformed_img = (
        np.where(image <= 0, 65536, image) if is_drop_non_positive else image
    )
    min_tansformed_img = (
        np.where(image <= 0, -65536, image) if is_drop_non_positive else image
    )

    truncated_lower = np.percentile(
        max_tansformed_img, truncated_percent, axis=(0, 1), keepdims=True
    )
    truncated_upper = np.percentile(
        min_tansformed_img, 100 - truncated_percent, axis=(0, 1), keepdims=True
    )

    stretched_img = (image - truncated_lower) / (truncated_upper - truncated_lower) * (
        stretch_range[1] - stretch_range[0]
    ) + stretch_range[0]
    stretched_img[stretched_img < stretch_range[0]] = stretch_range[0]
    stretched_img[stretched_img > stretch_range[1]] = stretch_range[1]
    if stretch_range[1] <= 255:
        stretched_img = np.uint8(stretched_img)
    elif stretch_range[1] <= 65535:
        stretched_img = np.uint16(stretched_img)
    return stretched_img

if __name__ == '__main__':
    # create_ICVL_valid()
    # create_ICVL_test()
    #Preprocess_apex_dataset()
    create_Urban_test()
    pass
