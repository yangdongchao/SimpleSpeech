from datasets import load_dataset, Audio
from transformers import EncodecModel, AutoProcessor
import pytorch_lightning as pl
from torchaudio.transforms import Resample
# dummy dataset, however you can swap this with an dataset on the 🤗 hub or bring your own
librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

# load the model + processor (for pre-processing the audio)
model = EncodecModel.from_pretrained("facebook/encodec_24khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

# cast the audio data to the correct sampling rate for the model
librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))
audio_sample = librispeech_dummy[0]["audio"]["array"]

# pre-process the inputs
inputs = processor(raw_audio=audio_sample, sampling_rate=processor.sampling_rate, return_tensors="pt")

# explicitly encode then decode the audio inputs
encoder_outputs = model.encode(inputs["input_values"], inputs["padding_mask"])
audio_values = model.decode(encoder_outputs.audio_codes, encoder_outputs.audio_scales, inputs["padding_mask"])[0]

# or the equivalent with a forward pass
audio_values = model(inputs["input_values"], inputs["padding_mask"]).audio_values

# you can also extract the discrete codebook representation for LM tasks
# output: concatenated tensor of all the representations
audio_codes = model(inputs["input_values"], inputs["padding_mask"]).audio_codes

class EncodecAE(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = EncodecModel.from_pretrained("facebook/encodec_24khz")

    def encode(self, x):
        return self.model.encode(x)
    
    def decode(self, x):
        return self.model.decode(x)

d_model = EncodecAE()
import torchaudio
import glob
import os
names = glob.glob('/home/jupyter/tmp_code/sgmse/source/*.wav')
for name in names:
    audio, sr = torchaudio.load(name)
    bs_name = os.path.basename(name)
    if sr != 24000:
        audio = Resample(sr, 24000)(audio)
    # print('audio ', audio.shape)
    audio = audio.unsqueeze(1)
    frames = d_model.model.encode(audio)
    print(frames)
    out = d_model.model.decode([frames, None])[:, :, :audio.shape[-1]]
    print(out.shape)
    assert 1==2
    # print('emb ', emb.shape)
    # codes = d_model.model.quantizer.encode(emb)
    # print('codes ', codes.shape)
    # codes = codes.transpose(0, 1) # 
    # print('codes 2 ', codes.shape)
    # emb = d_model.model.quantizer.decode(codes)
    # print('emb2 ', emb.shape)
    #out = d_model.model.decoder(emb)
    print('out ', out.shape)
    torchaudio.save('/home/jupyter/tmp_code/is2024/encodec/'+bs_name, out.squeeze(0), sample_rate=24000, encoding='PCM_S', bits_per_sample=16)
    assert 1==2


import argparse
import random
from functools import cache
from pathlib import Path

import soundfile
import torch
import torchaudio
from einops import rearrange
from encodec import EncodecModel
from encodec.utils import convert_audio
from torch import Tensor
from tqdm import tqdm


@cache
def _load_model(device="cuda"):
    # Instantiate a pretrained EnCodec model
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)
    model.to(device)
    return model


def unload_model():
    return _load_model.cache_clear()


@torch.inference_mode()
def decode(codes: Tensor, device="cuda"):
    """
    Args:
        codes: (b q t)
    """
    assert codes.dim() == 3
    model = _load_model(device)
    return model.decode([(codes, None)]), model.sample_rate


def decode_to_file(resps: Tensor, path: Path):
    assert resps.dim() == 2, f"Require shape (t q), but got {resps.shape}."
    resps = rearrange(resps, "t q -> 1 q t")
    wavs, sr = decode(resps)
    soundfile.write(str(path), wavs.cpu()[0, 0], sr)


def _replace_file_extension(path, suffix):
    return (path.parent / path.name.split(".")[0]).with_suffix(suffix)


@torch.inference_mode()
def encode(wav: Tensor, sr: int, device="cuda"):
    """
    Args:
        wav: (t)
        sr: int
    """
    model = _load_model(device)
    wav = wav.unsqueeze(0)
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.to(device)
    encoded_frames = model.encode(wav)
    qnt = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # (b q t)
    return qnt

def encode_from_file(path, device="cuda"):
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] == 2:
        wav = wav[:1]
    return encode(wav, sr, device)
