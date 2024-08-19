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
import dac
from torchaudio.transforms import Resample

class DACModel16k(nn.Module):
    def __init__(self, scalar_config, resume_path):
        super().__init__()
        model_path = dac.utils.download(model_type="16khz")
        self.model = dac.DAC.load(model_path)

    def encode(self, x):
        z, codes, latents, _, _ = self.model.encode(x)
        return z ,codes, latents
    
    def decode(self, z):
        return self.model.decode(z)


d_model = DACModel16k(None,None)
import torchaudio
import glob
import os
names = glob.glob('/home/jupyter/tmp_code/sgmse/source/*.wav')
for name in names:
    audio, sr = torchaudio.load(name)
    bs_name = os.path.basename(name)
    if sr != 16000:
        audio = Resample(sr, 16000)(audio)
    # print('audio ', audio.shape)
    z, codes, latents = d_model.encode(audio.unsqueeze(0))
    print('z ', z.shape, codes.shape)
    assert 1==2
    y = d_model.decode(z)
    print('y ', y.shape)
    torchaudio.save('/home/jupyter/tmp_code/is2024/dac_rec/'+bs_name, y.squeeze(0), sample_rate=16000, encoding='PCM_S', bits_per_sample=16)