import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from .binarize import binarize
from .method_inference import forward_haploid, forward_nadm, extend_if_lai




def cw_loss(logits, labels):
    labels_oh = torch.nn.functional.one_hot(labels, num_classes=logits.shape[1])
    if len(labels_oh.shape)== 3:
        labels_oh = labels_oh.permute(0,2,1)
    masked_logits = logits*labels_oh
    max_l, _ = torch.max(logits-masked_logits, dim=1, keepdims=True)
    logits_diff = torch.nn.functional.relu(masked_logits - max_l)
    return logits_diff.mean()



def subsample_adversarial_mask(bmask):
    acc_list = []
    adv_mask_per = []
    subsample_bmask_list = []
    betas = np.linspace(1.0, 0.0, num=20)# #[1.0, 0.75, 0.5, 0.25, 0.0]
    for beta in betas:
        subsample_bmask = bmask.clone()
        subsample_bmask = subsample_bmask*torch.bernoulli(torch.ones_like(bmask)*beta)
        subsample_bmask_list.append(subsample_bmask)
    return subsample_bmask_list, betas






def eval_net_and_subsample(bmask, x, y, net, forward_function, verbose=True):
    acc_list = []
    adv_mask_per = []
    subsample_bmask_list, betas = subsample_adversarial_mask(bmask)
    for subsample_bmask in subsample_bmask_list:
        x_d = torch.abs(x-subsample_bmask)
        out, pred = forward_function(net, x_d)
        y_ = extend_if_lai(y, out.shape)
        acc = torch.mean((pred.flatten() == y_.flatten()).float())
        if verbose:
            print('Accuracy eval is ', acc.item(), torch.mean(subsample_bmask))
        acc_list.append(acc.item())
        adv_mask_per.append(torch.mean(subsample_bmask).item())
    return acc_list, adv_mask_per




def generate_adversarial_mask(x,y,net,forward_function, lambda_c=0.1, num_iters=10, lr=0.1, verbose=True):
    
    # Moving input tensors to the same device as input network
    net_device = net.device
    x_device = x.device
    
    x = x.to(net_device)
    y = y.to(net_device)

    # Setting network into eval mode - make sure to freeze any batchnorm
    for module in net.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.eval()
        if hasattr(module,'with_grad'):
            module.with_grad=True
    
    
    # We use a version of C&W Loss
    criterion = cw_loss
    
    # Instanciate the latent sequence
    num_sequences, lenght_sequence = x.shape[0], x.shape[1]
    mask = torch.nn.Parameter((torch.randn(num_sequences, lenght_sequence)*0.1).float().to(x.device))

    # We use Adam - other optimizers could be used
    optimizer = optim.Adam([mask], lr=lr, weight_decay=0.0)
    

    for i in range(num_iters):
        # Obtain mask and perturbed sequence
        bmask = binarize(mask)
        x_d = torch.abs(x-bmask)
        
        # Forward the neural network
        out, pred = forward_function(net, x_d)

        # Compute loss and classification accuracy
        y_ = extend_if_lai(y, out.shape)
        loss = lambda_c*criterion(out, y_) + torch.mean(bmask) 
        acc = torch.mean((pred.flatten() == y_.flatten()).float())
        
        if verbose:
            # Print values if verbose
            print(i, 'mean ',  torch.mean(bmask).item(), 'loss ', loss.item(), 'acc ', acc.item())
            
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Return the final binary mask
    x = x.to(x_device)
    y = y.to(x_device)
    bmask = bmask.to(x_device)
    
    return bmask.detach()







def run_robustness_analysis(snps, labels, method_dict, bsize=64, store_dataset=False, device='cpu'):

    # Main function
    
    n_batch = int(snps.shape[0]/bsize)
    print(' A total of NBATCH ', n_batch)

    permidx = torch.randperm(snps.shape[0])
    snps, labels = snps[permidx,:], labels[permidx]

    real_seqs = []
    fake_seqs = []

    acc_list = []
    adv_mask_per_list = []
    for i in range(n_batch):
        print(i)
        s = i*bsize
        e = (i+1)*bsize
        x = torch.tensor(snps[s:e,:]).float().to(device)
        y = torch.tensor(labels[s:e]).long().to(device)



        bmask = generate_adversarial_mask(x, y, method_dict['model'], method_dict['forward'], num_iters=method_dict['n_adv_steps'], lambda_c=method_dict['lambda'], lr=method_dict['lr'])
        bmask_perm = bmask[:, torch.randperm(bmask.shape[1])]
        alphas = [1.0, 0.75, 0.5, 0.25, 0.0]

        if store_dataset:
            real_seqs.append(x.cpu().numpy())
            fake_seqs.append(torch.abs(x-bmask).cpu().numpy())

        acc_list_batch = []
        adv_mask_per_list_batch = []

        for alpha in alphas:
            b = torch.bernoulli(torch.ones_like(bmask)*alpha)
            bmask_rand = b*bmask + (1-b)*bmask_perm
            acc, adv_mask_per = eval_net_and_subsample(bmask_rand, x, y, method_dict['model'], method_dict['forward'])

            adv_mask_per_list_batch.append(adv_mask_per)
            acc_list_batch.append(acc)

        adv_mask_per_list.append(adv_mask_per_list_batch)
        acc_list.append(acc_list_batch)

    adv_per_array = np.array(adv_mask_per_list)
    acc_array = np.array(acc_list)
    
    return acc_array, adv_per_array, real_seqs, fake_seqs