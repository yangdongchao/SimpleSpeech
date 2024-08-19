import argparse, os, sys, glob
import pathlib
directory = pathlib.Path(os.getcwd())
print(directory)
sys.path.append(str(directory))
import torch
import numpy as np
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import pandas as pd
from tqdm import tqdm
from icecream import ic
from vocoder.bigvgan.models import VocoderBigVGAN
import soundfile
import torchaudio
from torchaudio.transforms import Resample
from preprocess.gpt_duration import get_struct
import time

def load_model_from_config(config, ckpt = None, verbose=True):
    model = instantiate_from_config(config.model)
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        sd = pl_sd["state_dict"]
        
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)
    else:
        print(f"Note chat no ckpt is loaded !!!")

    model.cuda()
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        default="A large truck driving by as an emergency siren wails and truck horn honks",
        help="the prompt to generate"
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default="16000",
        help="sample rate of wav"
    )
    parser.add_argument(
        "--test-dataset",
        default="none",
        help="test which dataset: audiocaps/clotho/fsd50k"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2audio-samples"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=100,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=32,
        help="latent height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=312,
        help="latent width, in pixel space",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for the given prompt",
    )

    parser.add_argument(
        "--test_file",
        type=str,
        default='',
        help="the prompt audio path",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=5.0, # if it's 1, only condition is taken into consideration
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        type=str,
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default="",
    )
    parser.add_argument(
        "--vocoder-ckpt",
        type=str,
        help="paths to vocoder checkpoint",
        default='vocoder/logs/audioset',
    )

    return parser.parse_args()

class GenSamples:
    def __init__(self,opt,sampler,model,outpath,vocoder = None,save_mel = True,save_wav = True) -> None:
        self.opt = opt
        self.sampler = sampler
        self.model = model
        self.outpath = outpath
        if save_wav:
            assert vocoder is not None
            self.vocoder = vocoder
        self.save_mel = save_mel
        self.save_wav = save_wav
        self.channel_dim = self.model.channels
    
    def gen_test_sample(self, prompt, duration, ref_audio=None, mel_name = None, wav_name = None, repaint=False):# prompt is {'ori_caption':’xxx‘,'struct_caption':'xxx'}
        uc = None
        record_dicts = []
        W = int(duration*16000/320)
        device = self.model.betas.device
        if self.opt.scale != 1.0:
            emptycap = {'ori_caption':self.opt.n_samples*[""]} # 
            uc = self.model.get_learned_conditioning(emptycap) #  get the unconditional embeding
            #print('uc ', uc.shape)
        for n in range(self.opt.n_iter):  # trange(self.opt.n_iter, desc="Sampling"):
            for k,v in prompt.items():
                # print(v)
                # assert 1==2
                prompt[k] = self.opt.n_samples * [v] # transfer to list type
            # print('prompt ', prompt)
            c = self.model.get_learned_conditioning(prompt)# shape:[1,77,1280],即还没有变成句子embedding，仍是每个单词的embedding
            #print('c ', c.shape)
            if self.channel_dim>0:
                shape = [self.channel_dim, self.opt.H, W]  # (z_dim, 80//2^x, 848//2^x)
            else:
                shape = [self.opt.H, W] # 32, 324
            # print('shape ', shape)
            # print('self.opt.ddim_steps ', self.opt.ddim_steps) 100
            # print('self.opt.n_samples ', self.opt.n_samples) 1
            # print('self.opt.n_iter ', self.opt.n_iter) 1
            if ref_audio is not None and repaint:
                if len(ref_audio.shape) == 2:
                    ref_audio = ref_audio.unsqueeze(0).to(device)
                latent_features = self.model.encode_first_stage(ref_audio)
                latent_features = latent_features.squeeze(0)
                #print('latent_features ', latent_features.shape)
                x0 = torch.randn(shape, device=device)
                mask = torch.ones(shape, device=device)
                mask[:,latent_features.shape[1]:] = 0
                x0[:,:latent_features.shape[1]] = latent_features # init the x0, mask
                # print('mask ', mask.shape, mask)
                # print('x0 ', x0.shape)
            else:
                mask = None
                x0 = None

            if ref_audio is not None and repaint == False:
                if len(ref_audio.shape) == 2:
                    ref_audio = ref_audio.unsqueeze(0).to(device)
                spk_embed = self.model.get_learned_spk_embed(ref_audio) # get the spk embedding


            samples_ddim, _ = self.sampler.sample(S=self.opt.ddim_steps,
                                                conditioning=[c,spk_embed],
                                                batch_size=self.opt.n_samples,
                                                shape=shape,
                                                verbose=False,
                                                mask=mask,
                                                x0=x0,
                                                unconditional_guidance_scale=self.opt.scale,
                                                unconditional_conditioning=[uc, spk_embed],
                                                eta=self.opt.ddim_eta)

            x_samples_ddim = self.model.decode_first_stage(samples_ddim)

            for idx,spec in enumerate(x_samples_ddim):
                spec = spec.squeeze(0).cpu().numpy()
                record_dict = {'caption':prompt['ori_caption'][0]}
                if self.save_mel:
                    mel_path = os.path.join(self.outpath, mel_name+f'_{idx}.npy')
                    np.save(mel_path,spec)
                    record_dict['mel_path'] = mel_path
                if self.save_wav:
                    wav = spec
                    wav_path = os.path.join(self.outpath,wav_name+f'.wav')
                    soundfile.write(wav_path, wav, self.opt.sample_rate)
                    record_dict['audio_path'] = wav_path
                record_dicts.append(record_dict)
        return record_dicts


def main():
    opt = parse_args()
    # print('opt ', opt)
    # assert 1==2
    config = OmegaConf.load(opt.base)
    # print("-------quick debug no load ckpt---------")
    # model = instantiate_from_config(config['model'])# for quick debug
    model = load_model_from_config(config, opt.resume)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    vocoder = VocoderBigVGAN(opt.vocoder_ckpt,device)
    #vocoder = None

    generator = GenSamples(opt, sampler, model, opt.outdir, vocoder, save_mel = False, save_wav = True)
    csv_dicts = []
    f = open(opt.test_file)
    with torch.no_grad():
        with model.ema_scope():
            st_time = time.time()
            for line in f:
                ans = line.strip().split('\t')
                print(ans)
                bs_name = ans[0]
                ori_caption = ans[3]
                prompt_audio = ans[1]
                ref_audio, sr = torchaudio.load(prompt_audio)
                sampling_rate = 16000
                if sr != sampling_rate:
                    ref_audio = Resample(sr, sampling_rate)(ref_audio)
                struct_caption = ori_caption
                duration = float(ans[-1]) 
                prompt = {'ori_caption': ori_caption,'struct_caption': struct_caption}
                generator.gen_test_sample(prompt, duration, ref_audio=ref_audio, mel_name=bs_name, wav_name=bs_name)
            print(time.time()-st_time)
    print(f"Your samples are ready and waiting four you here: \n{opt.outdir} \nEnjoy.")

if __name__ == "__main__":
    main()

