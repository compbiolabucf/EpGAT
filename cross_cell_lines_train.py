import argparse
import torch
from torch.autograd import Variable
from torch import nn, optim
from sklearn.model_selection import train_test_split
import os
import glob
import hicstraw
import pandas as pd
import numpy as np

from utils import *
from models import *

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)


def ccl_exp(args):
    res = args.bp_resolution
    window_size = args.window_length

    gpu = args.gpu
    epochs = args.epochs
    nb_heads = args.num_heads
    n_features = args.num_eps
    nb_embed = 2*n_features
    patience = args.patience
    learning_rate = args.learning_rate
    step_size = args.step_size
    gamma = args.gamma
    dropout = args.dropout

    chr = args.chromosome
    train_cell_line = args.train_cell_line
    test_cell_line = args.test_cell_line
    event = args.event
    l_hparams = args.loss_hyperparameters

    data_root = args.root_directory
    common_eps_fnm = f'{data_root}/{args.common_eps_file_name}.csv'
    common_eps = pd.read_csv(common_eps_fnm, index_col=0)['0']


    def get_data(cell_line, n_chr, num_eps):
        base_ends = {'chr1':246000000,
                    'chr2':240000000,
                    'chr3':198000000,
                    'chr4':186000000,
                    'chr5':180000000,
                    'chr6':168000000,
                    'chr7':150000000,
                    'chr8':144000000,
                    'chr9':138000000,
                    'chr10':132000000,
                    'chr11':132000000,
                    'chr12':132000000,
                    'chr13':114000000,
                    'chr14':102000000,
                    'chr15':96000000,
                    'chr16':90000000,
                    'chr17':78000000,
                    'chr18':78000000,
                    'chr19':54000000,
                    'chr20':60000000,
                    'chr21':42000000,
                    'chr22':48000000}

        chr = n_chr
        n_chr = 1 + n_chr
        adj_mats = []
        ep_mats = []
        inds = []
        preds = []
        gene_names = []

        count = 0

        chr_nm = f'chr{chr}'
        base_end = base_ends[chr_nm]

        #read data
        epi_data = pd.read_csv(f'{data_root}/{cell_line}/eps_data/{chr_nm}_eps.csv', index_col=0)
        epi_data = epi_data[common_eps]
        pred_data = pd.read_csv(f'{data_root}/{cell_line}/{event}_pred_data/{chr_nm}_pred.csv', index_col=0)
    
        print(f'\n\nAccessing data for {cell_line}:{chr_nm}\n\n')

        if os.path.exists(f'{data_root}/{cell_line}/{event}_adj_mats/{chr_nm}/'): 
            print('Using existing adjacency matrix')
            chr_files_exists = True 
        else: 
            print('Generating adjacency matrix from HiC file')
            chr_files_exists = False
            hic = hicstraw.HiCFile(f"{data_root}/{cell_line}.hic")

        start_pos = 0
        end_pos = window_size-1
        while end_pos<base_end:   
            
            ep_mat = epi_data.loc[epi_data.index<=int(end_pos/50)]
            ep_mat = ep_mat.loc[ep_mat.index>=int(start_pos/50)]
            ep_mat = np.log2(ep_mat+1)


            p_data = pred_data.loc[pred_data['End']<=end_pos]
            p_data = p_data.loc[p_data['Start']>=start_pos]


            if chr_files_exists: adj = sparse.load_npz(f'{data_root}/{cell_line}/{event}_adj_mats/{chr_nm}/{start_pos}_{end_pos}.npz').toarray()
            else: adj = get_mat_data(hic, data_root, cell_line, event, chr_nm, p_data.index, res, start_pos, end_pos)
            
            num_v = np.count_nonzero(adj)
            
            if num_v==0 or p_data.shape[0]<=2 or p_data['n/(N+n)'].sum() == 0: 
                start_pos += window_size
                end_pos = start_pos+window_size-1
                print(f'Found zero connections for window# {count} or not enough events.')
                continue
            
            
            ep_mats.append(Variable(torch.Tensor(np.array(ep_mat)).T))
            inds.append(Variable(torch.LongTensor(p_data.index-int(start_pos/200))))
            preds.append(Variable(torch.Tensor(np.array(p_data['n/(N+n)']))))
            adj_mats.append(Variable(torch.Tensor(adj)))


            gene_names.append(set(p_data['Gene']))
            start_pos += window_size
            end_pos = start_pos+window_size-1
            count += 1

        windows = np.arange(0,count,1)

        return ep_mats, inds, preds, adj_mats, windows, gene_names


    def out_avg(curr_pred, curr_out):
        pred_vals = set(list(curr_pred.reshape(-1)))
        for i in pred_vals:
            curr_out[curr_pred==i]=np.mean(curr_out[curr_pred==i])
        return curr_out




    print(f'-----------Exp for chromosome: {chr} ---------------')
    tr_mats, tr_inds, tr_preds, tr_adj_mats, tr_w, _ = get_data(train_cell_line, chr, n_features)
    eval_mats, eval_inds, eval_preds, eval_adj_mats, eval_w, eval_gene_nms = get_data(test_cell_line, chr, n_features)

    tr_w, val_w = train_test_split(tr_w, test_size=0.2, random_state=seed)


    # defining the model and loss function
    model = hic_model(in_channels=n_features, nb_embed=nb_embed, nb_heads=nb_heads, dropout=0.5)
    model = model.cuda(gpu)
    criterion1 = MLoss(l_hparams[0], l_hparams[1], l_hparams[2]).cuda(gpu)  



    def validate_model(model):
        ##validating
        model.eval()

        val_losses = []
        val_corrs = []
        for i in val_w:
           
            curr_ep = tr_mats[i].cuda(gpu)
            curr_adj = tr_adj_mats[i].cuda(gpu)
            curr_ind = tr_inds[i].cuda(gpu)
            curr_pred = tr_preds[i].cuda(gpu)

            output = model(curr_ep.reshape(1,n_features,-1), curr_adj, curr_ind)
            loss1 = criterion1(output, curr_pred.reshape(-1,1))
            
            loss_val = loss1

            curr_ep = tr_mats[i].detach().cpu()
            curr_adj = tr_adj_mats[i].detach().cpu()
            curr_ind = tr_inds[i].detach().cpu()
            curr_pred = tr_preds[i].detach().cpu()
            output = output.detach().cpu()

            val_loss = loss_val.item()

            
            torch.cuda.empty_cache()
            val_losses.append(val_loss)

            out_arr = output.squeeze().numpy()
            pred_arr = curr_pred.numpy()
            val_corrs.append(np.mean(np.corrcoef(out_arr, pred_arr)))

        avg_val_loss = np.mean(val_losses)
        print(f'Val loss: {avg_val_loss:.4f} | correlation: {np.mean(val_corrs)}')

        return avg_val_loss





    tr_l = []
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)   
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    prev_loss = 1e5

    draw_tr_plots = False
    loss_vals = []
    best = 1e10

    save_dir = f'ccl_saves/{event}/chr{chr}_saved_model'
    os.makedirs(save_dir, exist_ok=True)
    for epoch in range(epochs):        
        tr_loss = 0
        i = 0

        epoch_loss = 0
        
        for i in tr_w:
            optimizer.zero_grad()

            curr_ep = tr_mats[i].cuda(gpu)
            curr_adj = tr_adj_mats[i].cuda(gpu)
            curr_ind = tr_inds[i].cuda(gpu)
            curr_pred = tr_preds[i].cuda(gpu)


            output = model(curr_ep.reshape(1,n_features,-1), curr_adj, curr_ind)
            loss1 = criterion1(output, curr_pred.reshape(-1,1))
            loss_train = loss1
            
            loss_train.backward()
            optimizer.step()
            
            curr_ep = tr_mats[i].detach().cpu()
            curr_adj = tr_adj_mats[i].detach().cpu()
            curr_ind = tr_inds[i].detach().cpu()
            curr_pred = tr_preds[i].detach().cpu()
            output = output.detach().cpu()
            epoch_loss += loss_train.item()

            out_arr = output.squeeze().numpy()
            pred_arr = curr_pred.numpy()

        scheduler.step()

        mean_ep_loss = epoch_loss/tr_w.shape[0]
        print(f'\nEpoch: {epoch} | LR: {scheduler.get_last_lr()[0]:.4e} | train loss: {mean_ep_loss:.4f}')
        
        val_l = validate_model(model)
        loss_vals.append(val_l)


        tr_l.append(mean_ep_loss)
        torch.cuda.empty_cache()

        torch.save(model.state_dict(), f'{save_dir}/{epoch}.pkl')
        if loss_vals[-1] < best:
            best = loss_vals[-1]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == patience:
            break

        files = glob.glob(f'{save_dir}/*.pkl')
        for file in files:
            epoch_nb = int(file.split('/')[3].split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(file)

        
        
    print('------------------------------Training Ends-------------------------------')




    files = glob.glob(f'{save_dir}/*.pkl')
    for file in files:
        epoch_nb = int(file.split('/')[3].split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(file)

    # Restore best model
    print(f'Loading {best_epoch}th epoch')
    model.load_state_dict(torch.load(f'{save_dir}/{best_epoch}.pkl'))  #, weights_only=True))

    ##testing
    model.eval()

    corrs_lst = []
    loss_lst = []
    test_loss = 0
    test_corr = 0
    w_st = []
    w_en = []
    w_gene = []

    print('\n---Testing---')
    for i in eval_w:
        print('-----------------window# {}-----------------'.format(i))

        
        curr_ep = eval_mats[i].cuda(gpu)
        curr_adj = eval_adj_mats[i].cuda(gpu)
        curr_ind = eval_inds[i].cuda(gpu)
        curr_pred = eval_preds[i].cuda(gpu)
        
        output = model(curr_ep.reshape(1,n_features,-1), curr_adj, curr_ind)
        print('Output: \n', output.squeeze())
        print('True value: \n', curr_pred)

        torch.cuda.empty_cache()
        curr_ep = eval_mats[i].detach().cpu()
        curr_adj = eval_adj_mats[i].detach().cpu()
        curr_ind = eval_inds[i].detach().cpu()
        curr_pred = eval_preds[i].detach().cpu()
        output = output.detach().cpu()

        out_arr = output.squeeze().numpy()
        pred_arr = curr_pred.numpy()
        
        out_arr = out_avg(pred_arr, out_arr)
        curr_corr = np.mean(np.corrcoef(out_arr, pred_arr))
        test_corr += curr_corr
        corrs_lst.append(curr_corr)

        print('Correlation: ', curr_corr)
        w_st.append(i*window_size)
        w_en.append((i+1)*window_size-1)
        w_gene.append(eval_gene_nms[i])


    print('-----------------------------------------------------------------')
    print('Mean test correlation: ', test_corr/len(eval_w))


    corr_result = pd.DataFrame({'Start': w_st, 
                                'End': w_en,
                                'Correlation': corrs_lst, 
                                'Gene': w_gene}, index=eval_w )
    corr_result.sort_values(by='Correlation', inplace=True, ascending=False)
    corr_result.to_csv(f'ccl_saves/{event}/chr{chr}_result.csv')