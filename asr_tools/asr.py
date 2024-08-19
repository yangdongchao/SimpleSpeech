import whisper
import os
import glob
import time
import sys
import torch
import argparse
import logging
import tarfile
def get_parser():
    parser = argparse.ArgumentParser(
        description="convert a data list, do tokenization and save as a torch .pt file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-file", type=str, default=None, help="text file in the format <exampe_id> <content>")
    parser.add_argument("--rank", type=int, help="local GPU rank, if applicable")
    parser.add_argument("--output-file", type=str, default=None, help="text file in the format <exampe_id> <content>")
    return parser

def main(args):
    args = get_parser().parse_args(args)
    args.rank -= 1 # run.pl starts from 1 but the exact jobid / gpuid starts from 0   
    max_gpu = torch.cuda.device_count()
    args.rank = (args.rank % max_gpu) #
    device = torch.device(f"cuda:{args.rank}")
    model = whisper.load_model("base")
    model = model.to(device)
    f = open(args.output_file, 'w')
    f_in = open(args.input_file)
    for line in f_in:
        tmp = line.strip().split(' ')
        bs_name = tmp[0]
        name = tmp[1]
        result = model.transcribe(name)
        #bs_name = os.path.basename(name).replace('.wav','')
        f.write(bs_name+' '+result['text']+'\n')

if __name__ == "__main__":
    main(sys.argv[1:])
