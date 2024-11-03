# S2TDM: Spatial-Spectral Transformer-based Diffusion Model for Hyperspectral Image Denoising

This repository contains the code for the paper

<!-- [Arxiv]()  -->


## Installation
Clone this repository:

```
git clone git@github.com:zhehui-wu/S2TDM.git
```

The project was developed using Python 3.8.18, and torch 1.10.1. You can build the environment via pip as follow:

```
pip3 install -r requirements.txt
```
## Experiments

### Preparing the pretrained model and the datasets
1. Download the pretrained model from link: https://drive.google.com/drive/folders/1Q5kVov72_fJpfsoRZ91cfbEXxJZ93Y7U?usp=sharing, and put into ```checkpoints```.

2. Download ICVL dataset from link: https://icvl.cs.bgu.ac.il/hyperspectral/, APEX dataset from link: https://apex-esa.org/, and Urban dataset from link:  https://rslab.ut.ac.ir/data. 

3. Divide the ICVL dataset according to the divisions in ```data/ICVL_train.text```, ```data/ICVL_val.text```, and ```data/ICVL_test.text```.

4. Perform preprocessing operations such as cropping on the divided data set as follows:

```
python utils/lmdb_data.py
python utils/mat_data.py
```

### Testing
Denoising on ICVL dataset, and you can choose whether to save the intermediate results of the diffusion model by specify ```--save```

```
# Gaussian Denoising
python hsi_test.py -a ssdm -p ssdm_gaussian --test-dir /data/dataset/ICVL/Testing  -r -rp /checkpoints/model_gaussian.pth --image_size 512 --channels 31 --save --save-dir /results/

# Complex Denoising
python hsi_test.py -a ssdm -p ssdm_complex --test-dir /data/dataset/ICVL/Testing  -r -rp /checkpoints/model_complex.pth --image_size 512 --channels 31 --save --save-dir /results/
```

Denoising on Urban dataset

```
python hsi_urban_test.py -a ssdm_urban -p ssdm_urban --test-dir /data/dataset/Urban/Testing  -r -rp /checkpoints/model_urban.pth --image_size 256 --channels 210 --save --save-dir /results/
```


### Training
Training denoising model on ICVL dataset, and train 100, 25, and 25 epochs on ```ICVL64_31```, ```ICVL128_31```, and ```ICVL256_31``` respectively.

```
# Gaussian Denoising
python hsi_denoising_gauss.py -a ssdm -p ssdm_gaussian --train-dir /data/dataset/ICVL/Training/ICVL64_31.db --valid-dir /data/dataset/ICVL/Validating 

# Complex Denoising
python hsi_denoising_complex.py -a ssdm -p ssdm_complex --train-dir /data/dataset/ICVL/Training/ICVL64_31.db --valid-dir /data/dataset/ICVL/Validating 
```

Training denoising model on Urban dataset

```
python hsi_urban.py -a ssdm_urban -p ssdm_urban --train-dir /data/dataset/APEX/Training/APEX64_210.db 
```

## Citation and Acknowledgement

If you find our work useful in your research, please cite:

```

```

The codes are based on the repository of [QRNN3D](https://github.com/Vandermode/QRNN3D).
