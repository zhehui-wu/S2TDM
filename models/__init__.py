from .ssdm.arch import S2TDM
from .comparison_methods.hsid_cnn import HSID
from .comparison_methods.qrnn import QRNNREDC3D
from .comparison_methods.t3sc.multilayer import MultilayerModel
from .comparison_methods.nnet import NNet
from .comparison_methods.sst import SST
from .comparison_methods.sert import SERT
from .comparison_methods.unet import UNet

def ssdm():
    net = S2TDM(in_channel=62,out_channel=31,dim = 96,depths=[6,6,6],num_heads=[6,6,6])
    net.use_2dconv = True     
    net.bandwise = False 
    return net

def ssdm_urban():
    net = S2TDM(in_channel=420,out_channel=210,dim = 96*4,depths=[4,4,4],num_heads=[6,6,6])
    net.use_2dconv = True     
    net.bandwise = False          
    return net

def hsid():
    net = HSID(in_channels = 24)
    net.use_2dconv = True     
    net.bandwise = False 
    return net

def hsid_urban():
    net = HSID(in_channels = 120)
    net.use_2dconv = True     
    net.bandwise = False 
    return net

def qrnn3d():
    net = QRNNREDC3D(1, 16, 5, [1, 3], has_ad=True)
    net.use_2dconv = False
    net.bandwise = False
    return net

def t3sc():
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('models/competing_methods/T3SC/layers/t3sc.yaml')
    net = MultilayerModel(**cfg.params)
    net.use_2dconv = True
    net.bandwise = False
    return net

def t3sc_urban():
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('models/competing_methods/T3SC/layers/t3sc_urban.yaml')
    net = MultilayerModel(**cfg.params)
    net.use_2dconv = True
    net.bandwise = False
    return net

def nnet():
    net = NNet(1, 32, 5, 2)
    net.use_2dconv = False
    net.bandwise = False
    return net

def sst():
    net = SST(inp_channels=31,dim = 90, window_size=8,depths=[ 6,6,6,6,6,6],num_heads=[ 6,6,6,6,6,6],mlp_ratio=2)
    net.use_2dconv = True
    net.bandwise = False
    return net

def sst_urban():
    net = SST(inp_channels=210,dim = 210,window_size=8,depths=[ 6,6,6,6,6,6],num_heads=[ 6,6,6,6,6,6],mlp_ratio=2)
    net.use_2dconv = True
    net.bandwise = False
    return net

def sert():
    net = SERT(inp_channels=31,dim = 96, window_sizes=[16,32,32],depths=[ 6,6,6],num_heads=[ 6,6,6],split_sizes=[1,2,4],mlp_ratio=2,weight_factor=0.1,memory_blocks=128,down_rank=8) 
    net.use_2dconv = True     
    net.bandwise = False          
    return net

def sert_urban():
    net = SERT(inp_channels=210,dim = 96*2,window_sizes=[8,16,16],depths=[ 6,6,6],num_heads=[ 6,6,6],split_sizes=[2,4,4],mlp_ratio=2,down_rank=8,memory_blocks=128)  
    net.use_2dconv = True     
    net.bandwise = False          
    return net

def unet():
    net = UNet()
    net.use_2dconv = True     
    net.bandwise = False 
    return net