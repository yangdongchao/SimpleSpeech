# SimpleSpeech
This repository provides the open-source code for our InterSpeech 2024 paper SimpleSpeech and our latest paper SimpleSpeech 2. 

## News
- We fix the bug in scalar16k.py. and release two scalar codec models: 50dim_9scalar and 32dim_5scalar. 
```
wget https://huggingface.co/Dongchao/UniAudio/resolve/main/16k_50dim_9.zip
wget https://huggingface.co/Dongchao/UniAudio/resolve/main/16k_32dim_5.zip
```
**Note that we still try to clean the repository, this is not the finnal version.**

## Data 
download the your dataset for training. If you dataset does not includes text. Please first refer to asr_tools to get speech-text pair.

Then please use 
```
python scripts/get_tsv_tts.py 
```
to get the tsv file. And then, update the path for tsv in tts.yaml

## Training
```
bash run.sh
```
## Eval
```
bash eval.sh
```

## Acknowledgement
This codebase is based on our previous work Make-An-Audio https://github.com/Text-to-Audio/Make-An-Audio

## Reference
If you find this code is usefule for your research. Please cite

@article{simplespeech,
  title={SimpleSpeech: Towards Simple and Efficient Text-to-Speech with Scalar Latent Transformer Diffusion Models},
  author={Yang, Dongchao and Wang, Dingdong and Guo, Haohan and Chen, Xueyuan and Wu, Xixin and Meng, Helen},
  journal={Proc. INTERSPEECH},
  year={2024}
}
