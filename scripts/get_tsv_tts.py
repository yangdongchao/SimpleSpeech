import pandas as pd
import argparse
import glob
import os
# a simple example to construct tsv file.
# assume that we use LibriTTS dataset
name_ls = []
dataset_ls = []
ori_cap = []
audio_path = []
f_new = open('/home/jupyter/data/libritts_text.scp') # audio-text pair
new_dict = {}
for line in f_new:
    ans = line.strip().split('|')
    new_dict[ans[0]] = ans[1]

names = glob.glob("/home/jupyter/data/LibriTTS_R/train-other-500/*/*/*.wav") # pass each uttrance
for name in names:
    bs_name = os.path.basename(name).replace('.wav', '')
    if bs_name not in new_dict.keys():
        continue
    caps = new_dict[bs_name]
    # s_cap = di_cap[bs_name]
    name_ls.append(bs_name)
    dataset_ls.append('libritts')
    ori_cap.append(caps)
    # struct_cap.append(s_cap)
    audio_path.append(name)

#name	dataset	ori_cap	audio_path
a={'name': name_ls,'dataset': dataset_ls,'ori_cap': ori_cap,'audio_path': audio_path, 'caption': ori_cap}
df = pd.DataFrame(a)
df.to_csv('libritts_other_example.csv', sep='\t', index=False, header=['name', 'dataset', 'ori_cap', 'audio_path', 'caption'])
