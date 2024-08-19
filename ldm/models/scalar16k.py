import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio
from torch.nn.utils import weight_norm, remove_weight_norm
from torch.autograd.function import InplaceFunction
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm
import pytorch_lightning as pl
from omegaconf import OmegaConf
from collections import OrderedDict

def get_padding(kernel_size, dilation=1):
 
class scalar_func9(InplaceFunction):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return torch.round(9*input)/9
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class ScalarModel(nn.Module):
    def __init__(self, num_bands, sample_rate, causal, num_samples, downsample_factors, downsample_kernel_sizes,
                       upsample_factors, upsample_kernel_sizes, latent_hidden_dim, default_kernel_size,
                       delay_kernel_size, init_channel, res_kernel_size):
        super(ScalarModel, self).__init__()
        # self.args = args
        self.encoder = []
        self.decoder = []
        self.vq = scalar_func9() # using 9
        # Encoder parts
        self.encoder.append(
            weight_norm(
                Conv1d(
                    num_bands,
                    init_channel,
                    kernel_size=default_kernel_size,
                    causal=causal
                )
            )
        )
        if num_samples > 1:
            # Downsampling
            self.encoder.append(
                PreProcessor(init_channel,
                             init_channel,
                             num_samples,
                             kernel_size=default_kernel_size,
                             causal=causal))
        for i, down_factor in enumerate(downsample_factors):
            self.encoder.append(
                ResEncoderBlock(init_channel * np.power(2, i),
                                init_channel * np.power(2, i+1),
                                down_factor,
                                downsample_kernel_sizes[i],
                                res_kernel_size,
                                causal=causal))
        self.encoder.append(
            weight_norm(Conv1d(
                init_channel * np.power(2, len(downsample_factors)),
                latent_hidden_dim,
                kernel_size=default_kernel_size,
                causal=causal)))
        # Decoder
        # look ahead 
        self.decoder.append(
            weight_norm(Conv1d(
                latent_hidden_dim,
                init_channel * np.power(2, len(upsample_factors)),
                kernel_size=delay_kernel_size)))
        for i, upsample_factor in enumerate(upsample_factors):
            self.decoder.append(
                ResDecoderBlock(init_channel * np.power(2, len(upsample_factors) - i),
                             init_channel * np.power(2, len(upsample_factors) - i - 1),
                             upsample_factor,
                             upsample_kernel_sizes[i],
                             res_kernel_size,
                             causal=causal))
        if num_samples > 1:
            self.decoder.append(
                PostProcessor(
                    init_channel,
                    init_channel,
                    num_samples,
                    kernel_size=default_kernel_size,
                    causal=causal))
        self.decoder.append(
            weight_norm(Conv1d(
                init_channel,
                num_bands,
                kernel_size=default_kernel_size,
                causal=causal)))
        self.encoder = nn.ModuleList(self.encoder)
        self.decoder = nn.ModuleList(self.decoder)

    def forward(self, x):
        for i, layer in enumerate(self.encoder):
            if i != len(self.encoder) - 1:
                x = layer(x)
            else:
                x = F.tanh(layer(x))
        # import pdb; pdb.set_trace()
        x = self.vq.apply(x) # vq
        for i, layer in enumerate(self.decoder):
            x = layer(x)
        return x
    
    def encode(self, x):
        for i, layer in enumerate(self.encoder):
            if i != len(self.encoder) - 1:
                x = layer(x)
            else:
                x = F.tanh(layer(x)) # reverse to tanh
        emb = x 
        emb_quant = self.vq.apply(emb) # vq
        return emb
    
    def decode(self, x):
        x = self.vq.apply(x) # make sure the prediction follow the similar disctribution
        for i, layer in enumerate(self.decoder):
            x = layer(x)
        return x

    def inference(self, x):
        for i, layer in enumerate(self.encoder):
            if i != len(self.encoder) - 1:
                x = layer(x)
            else:
                x = F.tanh(layer(x)) # reverse to tanh
        emb = x 
        # import pdb; pdb.set_trace()
        emb_quant = self.vq.apply(emb) # vq
        x = emb_quant
        for i, layer in enumerate(self.decoder):
            x = layer(x)
        return emb, emb_quant, x


class ScalarAE(pl.LightningModule):
    def __init__(self, scalar_config, resume_path=None):
        super().__init__()
        self.scalar_config = scalar_config
        self.resume_path = resume_path
        exp_model_config = OmegaConf.load(self.scalar_config)
        self.model = ScalarModel(**exp_model_config.generator.config)
        if resume_path is not None:
            self.resume_model()

    def resume_model(self):
        parameter_dict = torch.load(self.resume_path)
        new_state_dict = OrderedDict()
        self.model.load_state_dict(parameter_dict['codec_model']) # load model
    
    def encode(self, x):
        return self.model.encode(x)
    
    def decode(self, x):
        return self.model.decode(x)

