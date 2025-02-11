import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
from ViT import ViT
import torch.nn.functional as F

class FeatureSelectModule(nn.Module):
    def __init__(self, image_size, input_size, num_classes, hidden_size, noisy_gating=True, C=1):
        super().__init__()  
        self.noisy_gating = noisy_gating
        self.output_size = num_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.C = C
        
        self.w_feature_select = nn.Parameter(torch.zeros(input_size, 1), requires_grad=True)
        self.w_feature_select_noise = nn.Parameter(torch.zeros(input_size, 1), requires_grad=True)

        self.softplus = nn.Softplus()
        
        self.feature_select_softmax = nn.Softmax(0)
        
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        
        self.vit = ViT("B_16_imagenet1k", image_size=image_size, num_classes=num_classes, pretrained=True)
        self.norm = nn.LayerNorm(self.input_size, eps=1e-6)
        self.fc = nn.Linear(self.input_size, self.output_size)
        self.indices = torch.randperm(int(196*0.1))

    def entropy(self, logits):
        probabilities = F.softmax(logits, dim=0)
        H = torch.sum(probabilities * torch.log(probabilities), dim=1)
        return H

    def cal_similiay(self, batch_token, top_k_indices):
        def _cal_similiay(token1, token2):
            similarity = F.cosine_similarity(token1, token2, dim=1)
            return similarity
        similarity = 0
        shuffled_indices = top_k_indices[self.indices]
        k = 3
        batch_shuffled_indices = torch.split(shuffled_indices, top_k_indices.size(0)//k, dim=0)
        for _batch_shuffled_indices in batch_shuffled_indices:
            for i in range(1, _batch_shuffled_indices.size(0)):
                similarity += _cal_similiay(batch_token[_batch_shuffled_indices[0]], batch_token[_batch_shuffled_indices[i]])
        return similarity/top_k_indices.size(0)
    
    
    def feature_select(self, ori_tokens, aug_tokens, patch_num, train, noise_epsilon=1e-2): 
        clean_logits = ori_tokens @ self.w_feature_select 
        if self.noisy_gating and train:
            raw_noise_stddev = ori_tokens @ self.w_feature_select_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev) 
            logits = noisy_logits
        else:
            logits = clean_logits
        logits = self.feature_select_softmax(logits)
                
        batch_token_list = []
        batch_logists = torch.split(logits, patch_num, dim=0)
        batch_tokens = torch.split(ori_tokens, patch_num, dim=0)
        
        aug_batch_tokens = torch.split(aug_tokens, patch_num, dim=0)
        
        contrast_loss = 0.0
        similiay_loss = 0.0
        entrop_loss   = 0.0
        
        for i in range(len(batch_tokens)):
            batch_token = batch_tokens[i]
            batch_logist = batch_logists[i]
            aug_batch_token = aug_batch_tokens[i]
            if self.C != 1:
                top_logits, top_indices = batch_logist.topk(int(self.C*patch_num)+1, dim=0)
            else:
                top_logits, top_indices = batch_logist.topk(int(self.C*patch_num), dim=0)
            top_k_logits = top_logits[:int(self.C*patch_num), :]
            top_k_indices = top_indices[:int(self.C*patch_num), :]
            top_k_gates = top_k_logits / (top_k_logits.sum(0, keepdim=True) + 1e-6)  # normalization
                
            zeros = torch.zeros_like(batch_logist, requires_grad=True) 
            norm_batch_logist = zeros.scatter(0, top_k_indices, top_k_gates)
                        
            #### contrast loss
            squared_diff = ((aug_batch_token-batch_token)*norm_batch_logist)**2
            sum_squared_diff = torch.sum(squared_diff)
            contrast_loss += torch.sqrt(sum_squared_diff)
            
            #### similiay_loss
            similiay_loss += self.cal_similiay(batch_token, top_k_indices[:int(0.1*patch_num), :])
            
            #### entrop_loss
            entrop_loss += self.entropy(norm_batch_logist)
            
            temp_token = (batch_token * norm_batch_logist).sum(dim=0, keepdim=True)
            batch_token_list.append(temp_token)
            break
        return batch_token_list, contrast_loss, similiay_loss, entrop_loss

    def forward(self, x, loss_coef=1e-2):     
        aug_x, origin_x = x[0].to("cuda:0"), x[1].to("cuda:0")
        aug_x = self.vit(aug_x)   
        origin_x = self.vit(origin_x)   
                
        patch_token = origin_x
        patch_token = patch_token.reshape(-1, patch_token.shape[-1]) ### (batch*patch_size, dim)

        aug_patch_token = aug_x
        aug_patch_token = aug_patch_token.reshape(-1, aug_patch_token.shape[-1]) ### (batch*patch_size, dim)
        
        #### feature select
        tokens_list_ori, contrast_loss, similiay_loss, entrop_loss = self.feature_select(patch_token, aug_patch_token, 196, self.training)
        select_feature_token_ori = torch.cat(tokens_list_ori, dim=0)
        select_feature_token_ori = self.norm(select_feature_token_ori)

        y = self.fc(select_feature_token_ori)
            
        return  y, similiay_loss[0], contrast_loss, entrop_loss[0]