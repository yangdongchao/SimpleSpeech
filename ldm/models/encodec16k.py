import sys
sys.path.append('/home/jupyter/code/StableTTS_v2/ldm/models')
from modules.seanet import SEANetEncoder, SEANetDecoder
from quantization  import ResidualVectorQuantizer
from typing import Sequence, Optional, Union
import math
import random
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from omegaconf import OmegaConf
from collections import OrderedDict
from torchaudio.transforms import Resample

class SoundStream(nn.Module):
    """ SoundStream model or EnCodec model.
    
    Args:
        n_filters (int): n_filters (int): Base width for the model.
        D (int): Intermediate representation dimension.
        target_bandwidths (Sequence[int]): Target bandwidths in K-bits/second.
        ratios (Sequence[int]): downsampling factors, whose multiplication is the hop size.
        sample_rate (int): wave sampling rate.
        bins (int): number of code words in a codebook.
        normalize (bool): audio normalization.

    """
    def __init__(
        self,
        n_filters: int = 32,
        D: int = 512,
        target_bandwidths: Sequence[Union[int, float]] = [0.5, 1, 1.5, 2, 4, 6],
        ratios: Sequence[int] = [8, 5, 4, 2], #  downsampling by 320
        sample_rate: int = 16000,
        bins: int = 1024,
        normalize: bool = False,
        causal: bool = False,
    ):
        super().__init__()
        self.hop_length = np.prod(ratios)
        # total nb of codebooks, e.g., 6Kb/s, sr=16000 and hop_length=320 => nq = 12
        n_q = int(1000 * target_bandwidths[-1] // (math.ceil(sample_rate / self.hop_length) * 10))
        self.frame_rate = math.ceil(sample_rate / np.prod(ratios)) # 50 Hz
        self.bits_per_codebook = int(math.log2(bins)) # 1024 => 10
        self.target_bandwidths = target_bandwidths
        self.n_q = n_q
        self.sample_rate = sample_rate

        # Encoder model
        self.encoder = SEANetEncoder(n_filters=n_filters, dimension=D, ratios=ratios, causal=causal)
        # RVQ model
        self.quantizer = ResidualVectorQuantizer(dimension=D, n_q=n_q, bins=bins)
        # Decoder model
        self.decoder = SEANetDecoder(n_filters= n_filters, dimension=D, ratios=ratios, causal=causal)
    
    def get_last_layer(self):
        return self.decoder.layers[-1].weight
    
    def forward(self, x: torch.Tensor):
        e = self.encoder(x)
        # randomly select a band-width during training
        bw = self.target_bandwidths[random.randint(0, len(self.target_bandwidths) - 1)] # [0, len(target_bandwidths) - 1], both included
        quantized, codes, bandwidth, commit_loss  = self.quantizer(e, self.frame_rate, bw)
        o = self.decoder(quantized)
        return o, commit_loss, None

    def encode(self, x: torch.Tensor, target_bw: Optional[int] = None) -> torch.Tensor:
        e = self.encoder(x)
        if target_bw is None:
            bw = self.target_bandwidths[-1]
        else:
            bw = target_bw
        quantized, codes, bandwidth, commit_loss  = self.quantizer(e, self.frame_rate, bw)
        return quantized

    def decode(self, quantized: torch.Tensor) -> torch.Tensor:
        o = self.decoder(quantized)
        return o

class EncodecAE(pl.LightningModule):
    def __init__(self, scalar_config, resume_path=None):
        super().__init__()
        self.scalar_config = scalar_config
        self.resume_path = resume_path
        exp_model_config = OmegaConf.load(self.scalar_config)
        self.model = SoundStream(**exp_model_config.generator.config)
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

