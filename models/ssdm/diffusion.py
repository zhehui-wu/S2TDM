from functools import partial
import torch
from torch import nn
import torch.nn.functional as F
from einops import reduce
from tqdm.auto import tqdm
# import sys
# sys.path.append("/home/wuzhehui/Hyper_CS/SERT-master")
from utils import common
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from torchvision import  utils
from pathlib import Path

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size = 64,
        channels = 31,
        timesteps = 2000,
        sampling_timesteps = 200,
        loss_type = 'l1',
        objective = 'pred_x0',
        beta_schedule = 'linear',
        p2_loss_weight_gamma = 1, # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1,
        ddim_sampling_eta = 1.
    ):
        super().__init__()

        self.model = model#Unet
        self.channels = channels

        self.image_size = image_size#128

        self.objective = objective#'pred_noise'

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'
        #beta
        if beta_schedule == 'linear':
            betas = common.linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = common.cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape#T
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type#L1

        # sampling related parameters
       
        self.sampling_timesteps = common.default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training
        
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta#1

        # helper function to register buffer from float64 to float32
   
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))#最小值为1e-20
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting

        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            common.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            common.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (common.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            common.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    #
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            common.extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            common.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = common.extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = common.extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    #
    def model_predictions(self, x,condition_x, t , clip_x_start = True):
        # 
        img = torch.cat((x,condition_x),dim = 1)
        model_output = self.model(img , t)#31
        # _,model_output = self.model(img , t)#31
        # x -> cat(x,x_codition)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else common.identity#

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)
    #pθ
    def p_mean_variance(self, x,condition_x,t , clip_denoised = True):
        preds = self.model_predictions(x, condition_x,t)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start
    #x（t）->x（t-1）
    @torch.no_grad()#
    def p_sample(self, x, condition_x,t: int, clip_denoised = True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, condition_x=condition_x,t = batched_times, clip_denoised = clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start
    #x（T）->x0
    @torch.no_grad()
    def p_sample_loop(self, shape , condition_img ,targets):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)

        x_start = None
        output = []
        x = []
        idx=0
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            img, x_start = self.p_sample(img, condition_img,t)

            band = condition_img.shape[-3]
            h,w = condition_img.shape[-2:]
            psnr = []
            for k in range(band):
                psnr.append(10*np.log10((h*w)/sum(sum((targets.squeeze().cpu().detach().numpy()[k]-x_start.squeeze().cpu().detach().numpy()[k])**2))))
            print(sum(psnr)/len(psnr))
            idx+=1
            x.append(idx)
            output.append(sum(psnr)/len(psnr))
            plt.xlabel("sampling step")
            plt.ylabel("PSNR")
            plt.plot(x, output)
            plt.savefig('/home/wuzhehui/Hyper_CS/test/1.png')
        
        return img

    @torch.no_grad()
    def ddim_sample(self, shape, condition_img, save, save_dir ,clip_denoised = True):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]


        img = torch.randn(shape, device = device)

        x_start = None
        idx = sampling_timesteps
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, condition_img,time_cond , clip_x_start = clip_denoised)

            if time_next < 0:
                img = x_start

            else:
                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]

                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()

                noise = torch.randn_like(img)
                img = x_start * alpha_next.sqrt() + \
                    c * pred_noise + \
                    sigma * noise

            idx -= 1
            if save:
                if self.channels == 31:
                    outputs = img[0,[28,15,9],:,:].unsqueeze(0)
                    utils.save_image(outputs, str(Path(save_dir) / f'sample-recon-{idx}.png'), nrow=1)
                
                if self.channels == 210:
                    outputs = img[0,[207,107,0],:,:].unsqueeze(0)
                    utils.save_image(outputs, str(Path(save_dir) / f'sample-recon-{idx}.png'), nrow=1)
            
            if time_next < 0:
                break

        return img

    @torch.no_grad()
    def sample(self , batches , img_size ,condition_img, save, save_dir):
        batch_size, channels = batches, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample

        return sample_fn((batch_size, channels, img_size, img_size), condition_img, save, save_dir)

    def q_sample(self, x_start, t, noise=None):
        noise = common.default(noise, lambda: torch.randn_like(x_start))
        return (
            common.extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            common.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    
    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, condition_img,t, noise = None):
        b, c, h, w = x_start.shape

        noise = common.default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        xt = self.q_sample(x_start = x_start, t = t, noise = noise)

        # predict and take gradient step
        x = torch.cat((xt,condition_img),dim = 1)
        model_out = self.model(x, t)
        # out1,out2 = self.model(x, t)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.loss_fn(model_out, target, reduction = 'none')
        # loss = self.loss_fn(out1, target, reduction = 'none') + self.loss_fn(out2, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * common.extract(self.p2_loss_weight, t, loss.shape)
        return loss.mean()
    
    def get_x2_bar_from_xt(self, x1_bar, xt, t):
        return (
                (xt - common.extract(self.sqrt_alphas_cumprod, t, x1_bar.shape) * x1_bar) /
                common.extract(self.sqrt_one_minus_alphas_cumprod, t, x1_bar.shape)
        )

    def forward(self, img , condition_img):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(img, condition_img,t)
    


