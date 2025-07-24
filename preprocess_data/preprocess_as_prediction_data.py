import os
import glob
import argparse
import pyBigWig
import numpy as np
import pandas as pd
from scipy import sparse



parser = argparse.ArgumentParser()
parser.add_argument('--pred_files_dir', type=str, default='raw_pred_data/')
parser.add_argument('--pred_file_name', type=str, default='AS_MCF7_mock_with_TX.xlsx')
parser.add_argument('--ep_files_dir', type=str, default='ep_data/')
parser.add_argument('--cell_line', type=str, default='MCF7')
parser.add_argument('--save_dir', type=str, default='../data')
parser.add_argument('--pred_resolution', type=int, default=200)
args = parser.parse_args()

save_dir = f'{args.save_dir}/{args.cell_line}/as_pred_data/'
os.makedirs(save_dir, exist_ok=True)

ep_files_dir = f'{args.ep_files_dir}/{args.cell_line}'
eps = glob.glob(f'{ep_files_dir}/*.bigWig')
check_ep = pyBigWig.open(eps[0])

pred_file = f'{args.pred_files_dir}/{args.cell_line}/{args.pred_file_name}'
out_data = pd.read_excel(pred_file)
pred_res = args.pred_resolution


def find_bins(curr_gene, arr):
    start = curr_gene['Exon Start']
    end = curr_gene['Exon End']
    
    st_bin = arr[out_arr[:,1]<=start][-1]
    end_bin = arr[out_arr[:,2]>=end][0]
    
    return arr[st_bin[0]:end_bin[0]+1]


for cnum in range(1,23):
    chr_nm = f'chr{cnum}'
    chr_len = check_ep.chroms(chr_nm)

    print(f'\nProcessing AS for {chr_nm}...')

    out_data.sort_values(by='Exon Start', inplace=True)
    out_chr = out_data.loc[out_data['Chrom']==chr_nm]
    out_chr['Length'] = out_chr['Exon End']-out_chr['Exon Start']

    bin_start = np.arange(1,chr_len, pred_res)
    bin_end = np.roll(bin_start-1,-1)
    bin_end[-1] = chr_len
    out_arr = np.stack((bin_start, bin_end), axis=1)
    bin_inds = np.arange(0,out_arr.shape[0],1)
    out_arr = np.stack((bin_inds, bin_start, bin_end), axis=1)

    valid_inds = []
    valid_starts = []
    valid_ends = []
    valid_as = []
    lengths = []
    gene_names = []

    for i in range(out_chr.shape[0]):
        gene = out_chr.iloc[i]
        curr = find_bins(gene, out_arr)

        for j in range(curr.shape[0]):
            valid_inds.append(curr[j][0])
            valid_starts.append(curr[j][1])
            valid_ends.append(curr[j][2])
            valid_as.append(gene['n1/(n1+N1)'])
            gene_names.append(gene['Gene Name'])
            lengths.append(gene['Length'])
            
            
    valid_data = pd.DataFrame([valid_starts,valid_ends,lengths,valid_as,gene_names]).T
    valid_data.index = valid_inds
    valid_data.columns = ['Start', 'End', 'Length','n/(N+n)', 'Gene']


    inds = []
    starts = []
    ends = []
    lengths = []
    ratios = []
    genes = []
    counter = 0

    print('Filtering proper reads....')
    for entry in set(valid_data.index):
        
        curr_entry = valid_data.loc[valid_data.index==entry]
        if curr_entry.shape[0]>1:
            curr_entry = curr_entry.loc[curr_entry['Length']==max(curr_entry['Length'])]
            # print('Inter: \n',curr_entry)
        if(curr_entry.shape[0]>1):
            curr_entry = curr_entry.loc[curr_entry['n/(N+n)']==max(curr_entry['n/(N+n)'])]
            # print('Final: \n',curr_entry)
            
        if(curr_entry.shape[0]>1):
            curr_entry = curr_entry.iloc[0]
            genes.append(curr_entry['Gene'])
            # print('Final+: \n',curr_entry)
        else: genes.append(curr_entry['Gene'].item())
            
        starts.append(curr_entry['Start'].item())
        ends.append(curr_entry['End'].item())
        lengths.append(curr_entry['Length'].item())
        ratios.append(curr_entry['n/(N+n)'].item())
        # genes.append(curr_entry['Gene'].item())
        inds.append(entry)
        
        counter += 1

    output_data = pd.DataFrame([starts,ends,lengths,ratios,genes]).T
    output_data.columns = ['Start', 'End', 'Length','n/(N+n)', 'Gene']
    output_data.index = inds

    output_data.to_csv(f'{save_dir}/{chr_nm}_pred.csv')


















