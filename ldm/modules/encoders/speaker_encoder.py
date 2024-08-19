import os
import glob
import torch
import tqdm
import transformers
import torchaudio
from torchaudio.transforms import Resample
import torch.nn as nn
from functools import partial
from ldm.util import count_params

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

class SpeakerEncoder(AbstractEncoder):
    def __init__(self, version="facebook/wav2vec2-large-xlsr-53", freeze=True, device="cuda"):  
        super().__init__()
        self.spk_emb = transformers.Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-large-xlsr-53").cuda()
        self.to(device)
        if freeze:
            self.freeze()

    def to(self, device):
        #self.t5_transformer.to(device)
        self.spk_emb.to(device)
        self.device = device

    def freeze(self):
        self.spk_emb = self.spk_emb.eval()
        for param in self.spk_emb.parameters():
            param.requires_grad = False
            param.grad = None
    
    def forward(self, wave):
        wave = wave.squeeze(1)
        outputs = self.spk_emb(wave, output_hidden_states=True)
        #spk_emb_global = outputs.hidden_states[1].mean(1).squeeze() #
        spk_emb =  outputs.hidden_states[1].mean(1) # get B, spk?
        # print('spk_emb ', spk_emb.shape)
        # assert 1==2
        return spk_emb
    