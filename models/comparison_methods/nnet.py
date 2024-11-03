# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class DConvB(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, inplace=False):
        super(DConvB, self).__init__()             
        self.xconv = nn.Sequential(nn.Conv3d(in_channels, channels//2, (1,k,k), 1, (0,1,1), bias=True), 
                                    nn.Conv3d(channels//2, channels//2, (k,1,1), 1, (1,0,0), bias=True),
                                    nn.Conv3d(channels//2, channels//2, (1,k,k), 1, (0,2,2), dilation=2, bias=True),
                                    nn.Conv3d(channels//2, channels,    (k,1,1), 1, (2,0,0), dilation=2, bias=True),
                                    nn.LeakyReLU(negative_slope=0.05, inplace=True))#nn.ReLU(inplace=True))#    
    def forward(self,x):       
        x = self.xconv(x)
        return x

class ConvB(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, inplace=False):
        super(ConvB, self).__init__()             
        self.xconv = nn.Sequential(nn.Conv3d(in_channels, channels//2, (1,k,k), s, (0,p,p), bias=True), 
                                    nn.Conv3d(channels//2, channels, (k,1,1), 1, (p,0,0), bias=True),
                                    nn.LeakyReLU(negative_slope=0.05, inplace=True))#nn.ReLU(inplace=True))#    
    def forward(self,x,noise_est=None):          
        x = self.xconv(x)
        return x

class DeConvB(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, inplace=False):
        super(DeConvB, self).__init__()               
        self.xconv =nn.Sequential(nn.ConvTranspose3d(in_channels, channels, k, s, p, bias=True),
                                    nn.LeakyReLU(negative_slope=0.05, inplace=True))#nn.ReLU(inplace=True))#
    def forward(self,x):              
        x = self.xconv(x)
        return x

class UpConvB(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, upsample=(1,2,2),inplace=False):
        super(UpConvB, self).__init__()            
        self.xconv =  nn.Sequential(nn.Upsample(scale_factor=upsample, mode='trilinear', align_corners=True),  
                                    nn.Conv3d(in_channels, channels, k, s, p, bias=True),
                                    nn.LeakyReLU(negative_slope=0.05, inplace=True))#
    def forward(self,x):              
        x = self.xconv(x)
        return x

class TransConvB(nn.Sequential):
    def __init__(self, in_channels, channels,k=3, inplace=False):
        super(TransConvB, self).__init__() 
        self.nconv = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                   nn.Conv3d(in_channels, channels//2, 1, 1, 0, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv3d(channels//2,in_channels, 1, 1, 0, bias=True),
                                   nn.Sigmoid())
        self.nconv1 = nn.Sequential(nn.Conv3d(in_channels, channels//2, (1,5,5), (1,2,2), (0,2,2), bias=True),
                                    nn.ReLU(inplace=True),
                                    nn.AdaptiveAvgPool3d((None,16,16)),
                                    nn.Conv3d(channels//2, channels//4, (1,3,3), (1,2,2), (0,1,1), bias=True),
                                    nn.ReLU(inplace=True),
                                    nn.AdaptiveAvgPool3d((None,1,1)),
                                    nn.Conv3d(channels//4,in_channels, 1, 1, 0, bias=True),
                                    nn.Sigmoid()
                                    )
    def forward(self,x):        
        n = self.nconv1(x)
        return n

class BootstrapConvB(nn.Sequential):
    def __init__(self, in_channels, channels,inplace=False):
        super(BootstrapConvB, self).__init__()        
        self.pconv1 = ConvB(channels, channels//2,k=5, p=2)
        self.pconv2 = ConvB(channels//2, channels//2,k=3)
        self.pconv3 = ConvB(channels//2, channels//2,k=1,p=0)

        self.xconv =nn.Sequential(nn.Conv3d((channels//2)*3, channels//2, 3, 1, 1, bias=True),
                                  nn.LeakyReLU(negative_slope=0.05, inplace=True),
                                  nn.Conv3d(channels//2, channels, 1, 1, 0, bias=True))

    def forward(self,x,noise_est):      
        xa = self.pconv1(noise_est*x)
        xb = self.pconv2(xa)
        xc = self.pconv3(xb)
        output = self.xconv(torch.cat((xa,xb,xc),dim=1))

        return output+x


class SparseQRNNLayer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, conv_layer, act='tanh'):
        super(SparseQRNNLayer, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.conv = conv_layer
        self.act = act
        self.dropout = nn.Dropout(p=0.2)

    def _conv_step(self, inputs):
        gates = self.conv(inputs)
        Z, F = gates.split(split_size=self.hidden_channels, dim=1)
        if self.act == 'tanh':
            return Z.tanh(), F.sigmoid()
        elif self.act == 'relu':
            return Z.relu(), F.sigmoid()
        elif self.act == 'none':
            return Z, F.sigmoid()
        else:
            raise NotImplementedError

    def _rnn_step(self, z, f, h):
        h_ = (1 - f) * z if h is None else f * h + (1 - f) * z
        return h_

    def forward(self, inputs, fname=None):        
        h = None 
        Z, F = self._conv_step(inputs)
        hsl = [] ; hsr = []
        for time, (z, f) in enumerate(zip(Z.split(1, 2), F.split(1, 2))):  # split along timestep            
            h = self._rnn_step(z, f, h)
            hsl.append(h)        
        h = None
        for time, (z, f) in enumerate(zip(reversed(Z.split(1, 2)), reversed(F.split(1, 2)))):  # split along timestep
            h = self._rnn_step(z, f, h)
            hsr.insert(0, h)        

        # return concatenated hidden states
        hsl = torch.cat(hsl, dim=2)
        hsr = torch.cat(hsr, dim=2)
        hs  = self._rnn_step(Z,F,None)
        hs = self.dropout(hs)

        if fname is not None:
            stats_dict = {'z':Z, 'f':F, 'hsl':hsl, 'hsr':hsr,'hs':hs}
            torch.save(stats_dict, fname)
        return hsl + hsr, hs

class SpectralNet(SparseQRNNLayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, act='tanh'):
        super(SpectralNet, self).__init__(
            in_channels, hidden_channels, DConvB(in_channels, hidden_channels*2), act=act)
            
            
class NNet(torch.nn.Module):
    def __init__(self, in_channels, channels, num_half_layer, downsample=None):
        super(NNet, self).__init__()
        
        self.spec_est_1 = SpectralNet(in_channels, channels)
        self.spec_est_2 = SpectralNet(channels, in_channels)
        self.spec_est_3 = TransConvB(in_channels, channels)

        self.spec_est_a = TransConvB(in_channels, channels)
        self.spec_est_b = TransConvB(in_channels, channels)
        
        self.nconv = nn.Conv3d(in_channels, channels, 1, 1, 0, bias=True)
        
        assert downsample is None or 0 < downsample <= num_half_layer
        interval = num_half_layer // downsample if downsample else num_half_layer+1

        self.feature_extractor = nn.Conv3d(in_channels, channels,3, 1, 1, bias=True) 
        # Encoder       
        self.encoder = nn.ModuleList()
        for i in range(1, num_half_layer+1):
            if i % interval:
                encoder_layer = BootstrapConvB(channels, channels)
            else:
                encoder_layer = ConvB(channels, channels, k=3, s=(1,2,2), p=1)
            self.encoder.append(encoder_layer)
        # Decoder
        self.decoder = nn.ModuleList()
        for i in reversed(range(1,num_half_layer+1)):
            if i % interval:                
                decoder_layer = DeConvB(channels, channels)
            else:
                decoder_layer = UpConvB(channels, channels)
            self.decoder.append(decoder_layer)

        self.reconstructor = nn.Conv3d(channels, in_channels,3, 1, 1, bias=True)

    def forward(self, x):
        hd1,hs1 = self.spec_est_1(x) 
        hd2,hs2 = self.spec_est_2(hd1+2*hs1)
        noise_est = self.spec_est_3(hd2+2*hs2) 
        
        noise_f = self.nconv(noise_est)
        
        num_half_layer = len(self.encoder)
        out = self.feature_extractor(x)
        xs=[out]
        for i in range(num_half_layer-1):
            out = self.encoder[i](out,noise_f)
            xs.append(out)
        out = self.encoder[-1](out,noise_f)
        for i in range(num_half_layer):
            out = self.decoder[i](out)
            out = out + xs.pop()
        out = self.reconstructor(out) 
        
        self.noise_est = noise_est
        self.noise_hs = self.spec_est_b(hs2)
        self.noise_hd = self.spec_est_a(hd2)
        
        return x-out
        