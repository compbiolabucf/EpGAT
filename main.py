import argparse
from cross_cell_lines_train import *
from cross_chromosomes_train import *
from cross_ccl_chr_train import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_type', type=str, default='cross_cell_line')
    ## cross cell line args
    parser.add_argument('--train_cell_line', type=str, default='MCF7')
    parser.add_argument('--test_cell_line', type=str, default='K562')
    parser.add_argument('--chromosome', type=int, default=1)
    ## cross chromosome args
    parser.add_argument('--cell_line', type=str, default='MCF7')
    parser.add_argument('--train_chrs', type=int, nargs='+', default=[1,2,3,4,5,8,9,10,11,12])
    parser.add_argument('--val_chrs', type=int, nargs='+', default=[6,13])
    parser.add_argument('--test_chrs', type=int, nargs='+', default=[7,14])
    ## common args
    parser.add_argument('--event', type=str, default='as')
    parser.add_argument('--common_eps_file_name', type=str, default='common_eps')
    parser.add_argument('--root_directory', type=str, default='data')
    parser.add_argument('--bp_resolution', type=int, default=200)
    parser.add_argument('--window_length', type=int, default=6000000)
    parser.add_argument('--gpu', type=str, default='cuda:0')
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_eps', type=int, default=32)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=3.5e-5)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--dropout', type=float, default=0.75)
    parser.add_argument('--loss_hyperparameters', type=float, nargs='+', default=[0.7,1,10**4])

    args = parser.parse_args()
    event_name = 'Alternative Splicing' if args.event=='as' else 'Alternative Polyadenylation'
    if args.exp_type == 'cross_cell_line': 
        print(f'\n-------Cross cell line exp, event: {event_name}--------')
        print(f'train and val cell line: {args.train_cell_line}, test cell line: {args.test_cell_line}')
        ccl_exp(args)
    elif args.exp_type == 'cross_chromosome': 
        print(f'\n-------Cross chromosome exp, cell line: {args.cell_line}, event: {args.event}-------')
        print(f'train chrs: {args.train_chrs}, val chrs: {args.val_chrs}, test chrs: {args.test_chrs}')
        cchrs_exp(args)
    elif args.exp_type == 'both':
        print(f'\n-------Cross chromosome cross cell line exp, event: {args.event}-------')
        print(f'train and val cell line: {args.train_cell_line}, test cell line: {args.test_cell_line}')
        print(f'train chrs: {args.train_chrs}, val chrs: {args.val_chrs}, test chrs: {args.test_chrs}')
        ccl_chrs_exp(args)
    else:
        print('Invalid experiment type')
        print('Please choose between: \'cross_cell_line\', \'cross_chromosome\', \'both\'')

if __name__=='__main__':
    main()