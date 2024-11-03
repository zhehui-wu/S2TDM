from tkinter import W
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from einops import rearrange
import  numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp* W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LePEAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0., qk_scale=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        if idx == 0:
           H_sp, W_sp = self.resolution, self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution, self.split_size
        else:
            print ("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_lepe(self, x, func):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp) ### B', C, H', W'

        lepe = func(x) ### B', C, H', W'
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp* self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, qkv,mask=None):
        """
        x: B L C
        """
        q,k,v = qkv[0], qkv[1], qkv[2]

        ### Img2Window
        H = W = self.resolution
        B, L, C = q.shape
       # assert L == H * W, "flatten img_tokens has wrong size"
      
        q = self.im2cswin(q)

       
        k = self.im2cswin(k)
        v, lepe = self.get_lepe(v, self.get_v)

        q = q * self.scale
        #print(q.shape,k.shape)
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
    
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe
        x = x.transpose(1, 2).reshape(-1, self.H_sp* self.W_sp, C)  # B head N N @ B head N C

        ### Window2Img
        x = windows2img(x, self.H_sp, self.W_sp, H, W).view(B, -1, C)  # B H' W' C

        return x

    def flops(self,shape):
        flops = 0
        H, W = shape
        #q, k, v = (B* H//H_sp * W//W_sp) heads H_sp*W_sp C//heads
        flops += ( (H//self.H_sp) * (W//self.W_sp)) *self.num_heads* (self.H_sp*self.W_sp)*(self.dim//self.num_heads)*(self.H_sp*self.W_sp)
        flops += ( (H//self.H_sp) * (W//self.W_sp)) *self.num_heads* (self.H_sp*self.W_sp)*(self.dim//self.num_heads)*(self.H_sp*self.W_sp)

        return flops


class RSAttention(nn.Module):
    """residual spectral attention (RSA)

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads
        bias (bool): If True, add a learnable bias to projection
    """
    def __init__(self, dim, num_heads, bias):
       
        super(RSAttention, self).__init__()
        self.num_heads = num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x):
        b,c,h,w = x.shape
        residual = x
        qkv = self.qkv(x)
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)

        out+=residual
        return out




class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=0, qk_scale=None, memory_blocks=128,down_rank=16,weight_factor=0.1,attn_drop=0., proj_drop=0.,split_size=1):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5  

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
       
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.weight_factor = weight_factor

        self.attns = nn.ModuleList([
                LePEAttention(
                    dim//2, resolution=self.window_size[0], idx = i,
                    split_size=split_size, num_heads=num_heads//2, dim_out=dim//2,
                    qk_scale=qk_scale, attn_drop=attn_drop)
                for i in range(2)])
        
        self.kernel = nn.Sequential(
            nn.Linear(dim*4, dim*2, bias=False),
        )

    def forward(self, x,mask=None):#conditon 16,384
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3)

        x1 = self.attns[0](qkv[:,:,:,:C//2],mask)
        x2 = self.attns[1](qkv[:,:,:,C//2:],mask)
        
        x = torch.cat([x1,x2], dim=2)
        x = rearrange(x, 'b n (g d) -> b n ( d g)', g=4)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class S2TLA(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,split_size=1,drop_path=0.0,
                 mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,  act_layer=nn.GELU):
        super(S2TLA,self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
            

        self.norm1 =  nn.LayerNorm(dim) 
        self.norm2 = nn.LayerNorm(dim) 
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.spatial_attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,split_size=split_size,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        self.spectral_attn = RSAttention(dim, num_heads, bias=False)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

        self.num_heads = num_heads


    def forward(self, x, t):

        B,C,H,W = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=1)#90 BC
        x = x.flatten(2).transpose(1, 2)
        shortcut = x
        x = self.norm1(x)
        x = modulate(x,shift_msa,scale_msa)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)  
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  
        
        attn_windows = self.spatial_attn(x_windows)
        
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        #spectral attention

        x = x.view(B, H * W, C)
        
        x = x.transpose(1, 2).view(B, C, H, W)

        x = self.spectral_attn(x)   #global spectral attention
        
        x = x.flatten(2).transpose(1, 2)

        # FFN
        x = shortcut + gate_msa.unsqueeze(1)*self.drop_path(x)#B,HW,C
        x = x + gate_mlp.unsqueeze(1)*self.drop_path(self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)))

        x = x.transpose(1, 2).view(B, C, H, W)
        
        return x


class RTBlock(nn.Module):
    def __init__(self,
        dim = 90,
        window_size=8,
        depth=6,
        num_head=6,
        mlp_ratio=2,
        qkv_bias=True, qk_scale=None,
        drop_path=0.0,
        split_size=1,
        ):
        super(RTBlock,self).__init__()
        self.blocks = nn.ModuleList([S2TLA(dim=dim,
                    input_resolution=window_size, 
                    num_heads=num_head, 
                    window_size=window_size,
                    shift_size=0 if i%2==0 else window_size//2,
                    split_size = split_size,
                    mlp_ratio=mlp_ratio,
                    drop_path = drop_path[i],
                    qkv_bias=qkv_bias, qk_scale=qk_scale,)
            for i in range(depth)])
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
    
    def forward(self,x,t):
        shourcut = x
        for block in self.blocks:
            x = block(x, t)
        out = self.conv(x)+shourcut
        return out
        
    
class S2TDM(nn.Module):
    def __init__(self, 
        in_channel = 62,
        out_channel = 31,
        dim = 96,
        depths = [6,6,6],
        num_heads = [6,6,6],
        window_sizes = [16,32,32],
        split_sizes = [1,2,4],
        mlp_ratio = 2,
        qkv_bias=True, qk_scale=None,
        bias=False,
        drop_path_rate=0.1,
    ):

        super(S2TDM, self).__init__()

        self.conv_first = nn.Conv2d(in_channel, dim, 3, 1, 1)
        self.num_layers = depths
        self.layers = nn.ModuleList()
        self.t_embedder = TimestepEmbedder(dim)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        for i_layer in range(len(self.num_layers)):
            layer = RTBlock(dim = dim,
            window_size=window_sizes[i_layer],
            depth=depths[i_layer],
            num_head=num_heads[i_layer],
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            split_size=split_sizes[i_layer],
            drop_path =dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])]
        )
            self.layers.append(layer)
        
        #self.fcn = FCN(num_input_channels=in_channel, num_output_channels=out_channel, dim=dim, num_hidden=[1,2,4,2,1])
            
        self.output = nn.Conv2d(int(dim), dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv_delasta = nn.Conv2d(dim,out_channel, 3, 1, 1)

        # self.feature = []

    def forward(self, img, time):
        _,condition = torch.chunk(img, 2, dim = 1)#31 
        t = self.t_embedder(time)
        x = self.conv_first(img)
        shortcut = x
        for layer in self.layers:
            x = layer(x,t)
            # self.feature.append(x)
        x = self.output(x + shortcut) 
        x = self.conv_delasta(x) + condition
        return x
    
###------------------time embedding--------------------###
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x
    
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class FCN(nn.Module):

    def __init__(self, num_input_channels=62, num_output_channels=31, dim = 96,num_hidden=[1, 2, 4, 2, 1]):
        super(FCN,self).__init__()
        
        self.linear_first = nn.Sequential(nn.Linear(num_input_channels, dim, bias=True),
                                          nn.GELU())
        self.model = nn.ModuleList()

        for i in range(len(num_hidden)-1):
            self.model.append(nn.Linear(dim*num_hidden[i], dim*num_hidden[i+1], bias=True))
            self.model.append(nn.GELU())

        self.linear_last = nn.Sequential(nn.Linear(dim*num_hidden[-1], num_output_channels))


        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 3 * dim, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
    def forward(self,A,time):
        _,A_condition = torch.chunk(A, 2, dim = 1)#B C/2 k
        A = A.permute(0,2,1)#BkC
        shift_linear, scale_linear, gate_linear = self.adaLN_modulation(time).chunk(3, dim=1)#90 BC
        A = self.linear_first(A)
        A = modulate(A, shift_linear, scale_linear)
        for linears in self.model:
            A = linears(A)
        A = gate_linear.unsqueeze(1)*A
        A = A_condition + self.linear_last(A).permute(0,2,1)

        return A


if __name__ == '__main__':
    from thop import profile, clever_format
    device = torch.device('cuda:0')
    x = torch.rand((1, 62, 64, 64)).to(device)
    #y = torch.rand((16, 31, 64, 64)).to(device)
    t = torch.randint(0, 1000, (1,), device=device).long()
    net = SSDM(depths=[ 6,6,6]).to(device)
    macs, params = profile(net, inputs=(x,t))
    macs, params = clever_format([macs, params], "%.4f")
    print(macs, params)
