import os
import argparse
import glob
import pyBigWig
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import sparse

parser = argparse.ArgumentParser()
parser.add_argument('--ep_files_dir', type=str, default='ep_data')
parser.add_argument('--cell_line', type=str, default='MCF7')
parser.add_argument('--save_dir', type=str, default='../data')
parser.add_argument('--ep_resolution', type=int, default=50)
args = parser.parse_args()

save_dir = f'{args.save_dir}/{args.cell_line}/eps_data/'
os.makedirs(save_dir, exist_ok=True)

ep_res = args.ep_resolution
input_files_dir = f'{args.ep_files_dir}/{args.cell_line}'
eps = glob.glob(f'{input_files_dir}/*.bigWig')
num_eps = len(eps)

print(f'Number of epigenetic signals: {num_eps}')
for chrom in range(21,23):
    chr_name = f'chr{chrom}'
    start = 0

    print(f'\nProcessing for {chr_name}')
    first=True
    ep_nms = []
    for ep in eps:
        ep_file = pyBigWig.open(ep)
        start=0
        chrom_len = ep_file.chroms()[chr_name]
        
        data = []
        while(start<chrom_len):
            if start+ep_res>chrom_len: end=chrom_len
            else: end=start+ep_res
            data.append(np.mean(ep_file.values(chr_name, start, end)))
            start += ep_res

        data = pd.DataFrame(data)
        if first: 
            chr_data = data
            first=False
        else: chr_data = pd.concat([chr_data, data], axis=1, join='inner')
        ep_nms.append(ep.split('/')[1].split('.')[0])\
        
    chr_data.columns = ep_nms
    chr_data.to_csv(f'{save_dir}/{chr_name}_eps.csv')