import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import dgl
from dgl import shortest_dist
import time
import random
import torch.backends.cudnn as cudnn
import argparse
import numpy as np
from dgl.nn import GATConv
from typing import Optional
import numpy as np
import gc
import torchmetrics
from transformers import BertModel, BertTokenizer,\
                         T5Tokenizer, T5EncoderModel,\
                         AlbertModel, AlbertTokenizer,\
                         XLNetModel, XLNetTokenizer,\
                         EsmTokenizer
import esm

seq_re_dir = {
    1: 'A', 2: 'R', 3: 'N', 4: 'D', 5: 'C',
    6: 'Q', 7: 'E', 8: 'G', 9: 'H', 10: 'I',
    11: 'L', 12: 'K', 13: 'M', 14: 'F', 15: 'P',
    16: 'S', 17: 'T', 18: 'W', 19: 'Y', 20: 'V'
}


def set_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def build_graph_from_distance_matrix(use_GNN, distance_threshold, distances, edgefeat):
    # 1A  3A  10A  
    weight_threshold = 1 / distance_threshold #The weight is the reciprocal: 0.1 Å corresponds to 10, 1 Å to 1, 10 Å to 0.1, and 50 Å to 0.02
    if use_GNN is True:
    # distances : (batch_size, sequence_length, sequence_length)
        batch_size, seq_len, _ = distances.shape
        graphs = []  
        all_edgefeat = []  
        all_distances = [] 
        device = distances.device
        
        for i in range(batch_size):
            distance = distances[i].to(device)  
            weight = torch.where(distance != 0.0, 1 / distance, torch.tensor(0.0, device=device))
            graph_matrix = torch.where(weight > weight_threshold, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))   
            src, dst = torch.nonzero(graph_matrix == 1.0, as_tuple=True)
            
            # DGL 
            graph = dgl.graph((src, dst), num_nodes=seq_len, device=device)
            graphs.append(graph)
            
            edgefeat_edges = edgefeat[i, src, dst].to(device)  
            distances_edges= distances[i, src, dst].to(device)  
            
            all_edgefeat.append(edgefeat_edges)
            all_distances.append(distances_edges)
            
            batched_graph = dgl.batch(graphs)
            
            batched_edgefeat = torch.cat(all_edgefeat, dim=0)
            batched_distances = torch.cat(all_distances, dim=0)
            batched_distances = batched_distances.unsqueeze(1)
    
    else:
        batch_size, seq_len, _ = distances.shape
        graphs = []  
        device = distances.device
        
        for i in range(batch_size):
            distance = distances[i].to(device)  
            weight = torch.where(distance != 0.0, 1 / distance, torch.tensor(0.0, device=device))
            graph_matrix = torch.where(weight > weight_threshold, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))  
            src, dst = torch.nonzero(graph_matrix == 1.0, as_tuple=True)
            graph = dgl.graph((src, dst), num_nodes=seq_len, device=device)
            graphs.append(graph)
            batched_graph = dgl.batch(graphs)

        batched_edgefeat = edgefeat
        batched_distances = distances.unsqueeze(-1)
    
    return batched_graph, batched_edgefeat 

def get_gpd_input(graphs,mask):

    subgraphs = dgl.unbatch(graphs)
    in_degrees_list = []
    out_degrees_list = []
    for subgraph in subgraphs:
        in_degrees_list.append(subgraph.in_degrees())
        out_degrees_list.append(subgraph.out_degrees())

    in_degrees_list = [subgraph.in_degrees() for subgraph in subgraphs]
    out_degrees_list = [subgraph.out_degrees() for subgraph in subgraphs]
    in_degree = pad_sequence(in_degrees_list, batch_first=True)
    out_degree = pad_sequence(out_degrees_list, batch_first=True)
    shortest_paths = []
    dist = []
    for subgraph in subgraphs:
        dist_matrix = shortest_dist(subgraph)
        shortest_paths.append(dist_matrix)
        dist = torch.cat(shortest_paths, dim=0)
    dist = dist.reshape(-1,mask.size(1),mask.size(1))
    #mask (batch_size, sequence_length)
    attn_mask = mask.unsqueeze(2) | mask.unsqueeze(1)#(batch_size, sequence_length, sequence_length)
    #attn_mask = mask.unsqueeze(1) | mask.unsqueeze(2)
    return in_degree, out_degree, attn_mask, dist
    #(batch_size,sequence_length)  (batch_size,sequence_length)  
    #(batch_size,sequence_length,sequence_length)  (batch_size,sequence_length,sequence_length) 

def get_lm_embedding(seqs_al:Tensor,tokenizer,pretrained_lm,use_lm):   
        if use_lm is True:
            freeze_layers = True# freeze lm parameters
            if freeze_layers:
                if isinstance(tokenizer, T5Tokenizer) and isinstance(pretrained_lm, T5EncoderModel):
                    for layer in pretrained_lm.encoder.block[:29]:
                        for param in layer.parameters():
                            param.requires_grad = False
                elif isinstance(tokenizer, BertTokenizer) and isinstance(pretrained_lm, BertModel):
                    modules = [pretrained_lm.embeddings, *pretrained_lm.encoder.layer[:29]]
                    for module in modules:
                        for param in module.parameters():
                            param.requires_grad = False
                elif isinstance(tokenizer, XLNetTokenizer) and isinstance(pretrained_lm, XLNetModel):
                    modules = [pretrained_lm.word_embedding, *pretrained_lm.layer[:29]]
                    for module in modules:
                        for param in module.parameters():
                            param.requires_grad = False
        seq_strings = []
        for seq in seqs_al:
            seq_string = ''.join([seq_re_dir[int(num.item())] for num in seq if int(num.item()) in seq_re_dir])
            seq_strings.append(seq_string)
        sequence = [' '.join(list(seq)) for seq in seq_strings]
        #print(sequence)
        #sequence = re.sub(r"[UZOB]", "X", sequence)
        ids = tokenizer(sequence, return_tensors='pt', padding=True, truncation=True,max_length=255)
        #encoded_input = tokenizer(sequence, return_tensors='pt', max_length=255)
        ids = {key: value.to(seqs_al.device) for key, value in ids.items()}
        input_ids = ids['input_ids']
        attention_mask = ids['attention_mask']
        #print(input_ids.shape)
        #print(attention_mask.shape)
        output= []
        lm_embedding= []
        if use_lm is True:
            with torch.no_grad():
                output = pretrained_lm(input_ids)
            #print(output.last_hidden_state.shape)
            lm_embedding = output.last_hidden_state
        else:
            seqs_al = seqs_al.long()
            with torch.no_grad():
                lm_embedding = F.one_hot(seqs_al, num_classes=1024)
        features = [] 
        max_len = 255 # sequence length
        for seq_num in range(len(lm_embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = lm_embedding[seq_num][:seq_len]
            features.append(seq_emd)
            max_len = max(max_len, seq_emd.size(0))

        padded_features = [F.pad(feature, (0, 0, 0, max_len - feature.size(0))) for feature in features]
        features_tensor = torch.stack(padded_features)
        
        del input_ids, attention_mask, output, lm_embedding, ids, seq_strings, sequence, features,padded_features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return features_tensor

def get_lm_embedding_(seqs_al: torch.Tensor, tokenizer, pretrained_lm, use_lm: bool):
    if use_lm:
        freeze_layers = True  
        if freeze_layers:
            if isinstance(tokenizer, T5Tokenizer) and isinstance(pretrained_lm, T5EncoderModel):
                for layer in pretrained_lm.encoder.block[:29]:
                    for param in layer.parameters():
                        param.requires_grad = False
            elif isinstance(tokenizer, BertTokenizer) and isinstance(pretrained_lm, BertModel):
                modules = [pretrained_lm.embeddings, *pretrained_lm.encoder.layer[:29]]
                for module in modules:
                    for param in module.parameters():
                        param.requires_grad = False
            elif isinstance(tokenizer, XLNetTokenizer) and isinstance(pretrained_lm, XLNetModel):
                modules = [pretrained_lm.word_embedding, *pretrained_lm.layer[:29]]
                for module in modules:
                    for param in module.parameters():
                        param.requires_grad = False
        #"""
    #print(f"Tokenizer type: {type(tokenizer)}")
    if isinstance(tokenizer, EsmTokenizer):
        seqs = seqs_al
        sequences = seqs
        seq_strings = []
        for seq in sequences:
            seq_string = ''.join([seq_re_dir[int(num.item())] for num in seq if int(num.item()) in seq_re_dir])
            seq_strings.append(seq_string)
        sequences = [''.join(list(seq)) for seq in seq_strings]
        ids = tokenizer(sequences, padding='max_length', truncation=True, max_length=100, return_tensors='pt')
        ids = {key: value.to(seqs.device) for key, value in ids.items()}

        input_ids = ids['input_ids'].to(pretrained_lm.device)  # input_ids
        attention_mask = ids['attention_mask'].to(pretrained_lm.device)  # attention_mask
        
        # BERT
        with torch.no_grad():
            bert_output = pretrained_lm(input_ids=input_ids, attention_mask=attention_mask)
        
        # logits
        linear_layer = nn.Linear(bert_output["logits"].size(1),1024).to(seqs_al.device)
        output_feature = linear_layer(bert_output["logits"]).to(seqs_al.device)
        output_feature = output_feature.unsqueeze(1).repeat(1, seqs_al.size(1), 1)
        features_tensor = output_feature.to(seqs_al.device)
    #"""
    
    #"""
    # one-hot 1024
    else:
        seq_strings = []
        for seq in seqs_al:
            seq_string = ''.join([seq_re_dir[int(num.item())] for num in seq if int(num.item()) in seq_re_dir])
            seq_strings.append(seq_string)
        sequence = [' '.join(list(seq)) for seq in seq_strings]
        #print(sequence)
        #sequence = re.sub(r"[UZOB]", "X", sequence)
        ids = tokenizer(sequence, return_tensors='pt', padding=True, truncation=True,max_length=100)
        #encoded_input = tokenizer(sequence, return_tensors='pt', max_length=255)
        ids = {key: value.to(seqs_al.device) for key, value in ids.items()}
        input_ids = ids['input_ids']
        attention_mask = ids['attention_mask']
        #print(input_ids.shape)
        #print(attention_mask.shape)
        output= []
        lm_embedding= []
        if use_lm is True:
            with torch.no_grad():
                output = pretrained_lm(input_ids)
                #print(output.last_hidden_state.shape)
                lm_embedding = output.last_hidden_state
        else:
            #print("Hey! I am hot")
            seqs_al = seqs_al.long()
            with torch.no_grad():
                lm_embedding = F.one_hot(seqs_al, num_classes=1024)
        features = [] 
        max_len = 100 # sequence length
        for seq_num in range(len(lm_embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = lm_embedding[seq_num][:seq_len]
            features.append(seq_emd)
            max_len = max(max_len, seq_emd.size(0))

        padded_features = [F.pad(feature, (0, 0, 0, max_len - feature.size(0))) for feature in features]
        features_tensor = torch.stack(padded_features)
    #"""

    #del input_ids, attention_mask, output, lm_embedding, ids, seq_strings, sequence, features, padded_features
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return features_tensor

def accuracy(predict, label): 
    device = predict.device  
    label = label.to(device)
    accuracy_metric = torchmetrics.Accuracy(task="binary").to(device)
    predict = (predict > 0.5).float() 
    return accuracy_metric(predict, label).item()

def precision(predict, label):  
    device = predict.device  
    label = label.to(device)
    precision_metric = torchmetrics.Precision(task="binary").to(device)
    predict = (predict > 0.5).float()  
    return precision_metric(predict, label).item()

def specificity(predict, label):  
    device = predict.device  
    label = label.to(device)
    specificity_metric = torchmetrics.Specificity(task="binary").to(device)
    predict = (predict > 0.5).float()  
    return specificity_metric(predict, label).item()

def recall(predict, label):  
    device = predict.device  
    label = label.to(device)
    recall_metric = torchmetrics.Recall(task="binary").to(device)
    predict = (predict > 0.5).float()  
    return recall_metric(predict, label).item()

def f1_score(predict, label):  # F1
    device = predict.device  
    label = label.to(device)
    f1_metric = torchmetrics.F1Score(task="binary").to(device)
    predict = (predict > 0.5).float()  
    return f1_metric(predict, label).item()

def auc(predict, label):  # ROC
    device = predict.device  
    label = label.to(device)
    auc_metric = torchmetrics.AUROC(task="binary").to(device)
    return auc_metric(predict, label).item()

def mcc(predict, label):  
    device = predict.device  
    label = label.to(device)
    mcc_metric = torchmetrics.MatthewsCorrCoef(task="binary").to(device)
    predict = (predict > 0.5).float()  
    return mcc_metric(predict, label).item()

