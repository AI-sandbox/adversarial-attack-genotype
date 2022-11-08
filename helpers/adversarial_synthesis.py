import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from .binarize import binarize
from .method_inference import forward_haploid, forward_nadm, extend_if_lai, compute_pca, unique_to_percentage


def generate_adversarial_population(num_sequences, lenght_sequence, train_snps, train_labels, pca, target_label_pca, lainet, target_label_lainet, knn_W, target_knn_label, nadm, target_label_nadm, device, num_iters = 200, verbose=True, lr = 0.001):
    
    (W, b) = pca
    _, mean_pca_target, cov_pca_target, std_pcs_target = compute_pca(torch.tensor(train_snps)[torch.tensor(train_labels).long()==target_label_pca,:], W, b)

    for module in lainet.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.eval()
        if hasattr(module,'with_grad'):
            module.with_grad=True
            
    for module in nadm.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.eval()
        if hasattr(module,'with_grad'):
            module.with_grad=True
            
    
    mean_pca_target, cov_pca_target, std_pcs_target = mean_pca_target.to(device), cov_pca_target.to(device), std_pcs_target.to(device)
    
    lainet = lainet.to(device)
    nadm = nadm.to(device)
    W, b = W.to(device), b.to(device)
    knn_templates = torch.matmul(torch.tensor(train_snps[train_labels==target_knn_label,:]).float(), knn_W).to(device)
    knn_W = knn_W.to(device)
    
    centroid = torch.tensor(np.mean(train_snps[train_labels==target_label_pca,:], axis=0)).float().to(device)
    mask = torch.nn.Parameter(torch.randn(num_sequences,lenght_sequence).to(device)*0.5+(centroid.repeat(num_sequences,1)-0.5))

    optimizer = optim.Adam([mask], lr=lr, weight_decay=0.0)
    
    
    for i in range(num_iters):
        bmask = binarize(mask)
        pcs, mean_pcs, cov_pcs, std_pcs = compute_pca(bmask, W, b)
        
        out_lainet, pred_ln = forward_haploid(lainet, bmask)
        out_nadm, pred_nadm = forward_nadm(nadm, bmask)
        
        
        tensor_target_lainet = torch.tensor(target_label_lainet).unsqueeze(0).repeat(out_lainet.shape[0]).to(device)
        tensor_target_nadm = torch.tensor(target_label_nadm).unsqueeze(0).repeat(out_lainet.shape[0]).to(device)
        tensor_target_lainet = extend_if_lai(tensor_target_lainet, out_lainet.shape)

        loss_lainet = torch.nn.CrossEntropyLoss()(out_lainet, tensor_target_lainet)  
        loss_nadm = torch.nn.CrossEntropyLoss()(torch.log(out_nadm), tensor_target_nadm) 
        loss_knn = torch.mean(torch.square(torch.matmul(bmask,knn_W) - knn_templates[0:bmask.shape[0],:]))
        loss_pca = torch.mean(torch.square(pcs - mean_pca_target)) + torch.mean(torch.square(std_pcs - std_pcs_target))
        loss = loss_pca + loss_lainet + 2*loss_nadm + loss_knn 

        
        vm_ln, _ = torch.mode(pred_ln.flatten())
        vm_nadm, _ = torch.mode(pred_nadm.flatten())
        
        if verbose:
            print(i, loss.item(), vm_ln, vm_nadm, loss_pca.item(), loss_lainet.item(), loss_nadm.item(), loss_knn.item())


        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    return bmask.detach()