import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from transformers import T5Tokenizer, T5EncoderModel
from typing import Optional
import numpy as np
from utils import *

np.set_printoptions(threshold=np.inf)
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class ResTransformer(nn.Module):
    def __init__(self, embed_dim):
        super(ResTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.dense_dim = embed_dim*8
        self.attention_layer = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, dropout=0.05)
        self.fc = nn.Sequential(
        nn.Linear(self.embed_dim, self.dense_dim),  
        nn.ReLU(),                        
        nn.Linear(self.dense_dim, self.embed_dim)   
        )
        self.fc_bn_1 = nn.LayerNorm(embed_dim)
        self.fc_bn_2 = nn.LayerNorm(embed_dim)

        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(d_model=embed_dim, nhead=8, batch_first=True),
            num_layers=1 
        )
    def forward(self, x):
        #"""
        #transformer Encoder
        # [B, 8, 8, 64]
        x = x.permute(0, 2, 3, 1)  # [B, H=8, W=8, C=64]
        # （B, N, C）
        B, H, W, C = x.shape
        x_flat = x.view(B, H * W, C)  # [B, 64, 64]
        # Transformer
        x_enc,_ = self.attention_layer(x_flat,x_flat,x_flat)  # [B, 64, 64]
        x_enc = x_enc+x_flat
        x_input = self.fc_bn_1(x_enc)#(batch_size,255,60)
        x_output = self.fc_bn_2(x_input + self.fc(x_input))##(batch_size,255,60)
        x_enc = x_output

        #x_enc = self.transformer(x_flat)
        x_out = x_enc.view(B, H, W, C)  # [B, 8, 8, 64]
        x = x_out.permute(0, 3, 1, 2)   # [B, 64, 8, 8]
        return x
        #transformer Encoder


# conv + BN + ReLU 
class ConvBnReLU2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)
    
#  (B, 100, 100, 8)
class CostRegNet(nn.Module):
    def __init__(self):
        super(CostRegNet, self).__init__()
        self.conv0 = nn.Sequential(
        nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False),
        ConvBnReLU2D(8, 8))
        
        self.conv1 = ConvBnReLU2D(8, 16, stride=2)
        self.conv2 = ConvBnReLU2D(16, 16)

        self.conv3 = ConvBnReLU2D(16, 32, stride=2)
        self.conv4 = ConvBnReLU2D(32, 32)

        self.conv5 = ConvBnReLU2D(32, 64, stride=2)
        self.conv6 = ConvBnReLU2D(64, 64)

        self.res_64 = ResTransformer(64)
        self.res_32 = ResTransformer(32)
        self.res_16 = ResTransformer(16)
        self.res_8 = ResTransformer(8)

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True))
        
        self.final_upsample = nn.Upsample(size=(100, 100), mode='bilinear', align_corners=False)

    def forward(self, x):
        # input：[B, 100, 100, 8] -> [B, 8, 100, 100]
        x = x.permute(0, 3, 1, 2)

        conv0 = self.conv0(x)  # [B, 8, 64, 64]
        conv2 = self.conv2(self.conv1(conv0))  # [B, 16, 32, 32]
        conv4 = self.conv4(self.conv3(conv2))  # [B, 32, 16, 16]
        x = self.conv6(self.conv5(conv4))      # [B, 64, 8, 8]

        #x = self.res_64(x)

        x = conv4 + self.deconv1(x)            # [B, 32, 16, 16]
        x = conv2 + self.deconv2(x)            # [B, 16, 32, 32]
        x = conv0 + self.deconv3(x)            # [B, 8, 64, 64]

        x = self.final_upsample(x)             # → [B, 8, 100, 100]
        # output: [B, 8, 100, 100] -> [B, 100, 100, 8]
        return x.permute(0, 2, 3, 1)

class ABP_Text_Picture_Model(nn.Module):
    def __init__(self, node_emb_size=1024, num_heads=8, edge_feats=8):
        super(ABP_Text_Picture_Model, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = 128   #128
        self.dense_dim = 512  #1024
        self.efun = CostRegNet()
        self.preprocess = nn.Sequential(
            nn.Linear(1031, self.embed_dim),  
            nn.ReLU()
        )
        
        # DSSP embedding
        self.embedding_DSSP = nn.Embedding(9, 3)

        self.node_attention_layer = nn.MultiheadAttention(
            embed_dim=self.embed_dim, 
            num_heads=num_heads, 
            dropout=0.05
        )
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=self.embed_dim, 
            num_heads=num_heads, 
            dropout=0.05
        )

        self.fc = nn.Sequential(
        nn.Linear(self.embed_dim, self.dense_dim),  
        nn.ReLU(),                        
        nn.Dropout(p=0.2), 
        nn.Linear(self.dense_dim, self.embed_dim)   
        )
        self.fc_norm1 = nn.LayerNorm(self.embed_dim)
        self.fc_norm2 = nn.LayerNorm(self.embed_dim)

        self.edge_fc = nn.Sequential(
        nn.Linear(8,128),
        nn.ReLU(),
        nn.Linear(128,8)
        )
        self.edge_fc_norm1 = nn.LayerNorm(8)
        self.edge_fc_norm2 = nn.LayerNorm(8)

        # Modality projection layer - maps different modalities into a shared representation space
        self.node_projector = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128)
        )
        
        self.edge_projector = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128)
        )
        
        # Cross modal attention layer
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=8,
            dropout=0.1
        )
        
        # Modality fusion layer
        self.modal_fusion = nn.Sequential(
            nn.Linear(256, 256),  # Input after concatenating two 128-dimensional feature vectors
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.embed_dim)
        )
        
        # Temperature parameter for contrastive learning
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, 1),
            nn.Sigmoid()
        )

    def contrastive_loss(self, node_features, edge_features):
        """
            node_features: (batch_size * seq_len, 128) 
            edge_features: (batch_size * seq_len * seq_len, 128) 
        """
        # L2 
        node_features = F.normalize(node_features, dim=1)
        edge_features = F.normalize(edge_features, dim=1)
        
        # Similarity matrix
        similarity_matrix = torch.matmul(node_features, edge_features.T) / self.temperature
        
        # Labels
        labels = torch.arange(similarity_matrix.size(0)).to(similarity_matrix.device)
        
        # contrastive loss
        loss_i2t = F.cross_entropy(similarity_matrix, labels)
        loss_t2i = F.cross_entropy(similarity_matrix.T, labels)
        
        return (loss_i2t + loss_t2i) / 2

    def multimodal_alignment(self, node_feat, edge_feat, batch_size, seq_len, distance_matrix=None, distance_threshold=10.0):
        """
        Multimodal feature alignment and fusion (with neighbor filtering mechanism)
        Args:
        node_feat: (N, 128) node features
        edge_feat: (N, N, 8) edge feature matrix
        batch_size: batch size
        seq_len: sequence length
        distance_matrix: (batch_size, seq_len, seq_len) distance matrix used for neighbor filtering
        distance_threshold: distance threshold (default 10.0 Å); only neighbors with distance < threshold are aggregated

        Returns:
        fused_features: (N, 128) fused features
        contrastive_loss: contrastive learning loss
        """
        
        # 1.Project features into a shared space
        node_proj = self.node_projector(node_feat)  # (N, 128)
        
        # Get the last dimension of edge_feat
        edge_feat_dim = edge_feat.shape[-1]
        #print(f"edge_feat last dimension: {edge_feat_dim}")
        
        # Reshape and project edge features
        edge_feat_reshaped = edge_feat.view(-1, edge_feat_dim)
        #print(f"edge_feat_reshaped shape: {edge_feat_reshaped.shape}")
        
        edge_proj = self.edge_projector(edge_feat_reshaped)
        #print(f"edge_proj shape: {edge_proj.shape}")
        
        # 2. Compute correct reshape dimensions
        # edge_proj should have shape (N*N, 128), and we need to reshape it to (N, N, 128)
        total_edges = edge_proj.size(0)
        num_nodes = node_feat.size(0)
        
        if total_edges % num_nodes != 0:
            print(f"Warning: total_edges ({total_edges}) not divisible by num_nodes ({num_nodes})")
            edge_proj_for_pooling = edge_proj.view(num_nodes, -1, 128)
        else:
            edges_per_node = total_edges // num_nodes
            edge_proj_for_pooling = edge_proj.view(num_nodes, edges_per_node, 128)

        # Check if it can be reshaped without mismatch
        if distance_matrix is not None:
            # Flatten the distance matrix and reshape it to (batch_size*seq_len, seq_len, 1)
            distance_flat = distance_matrix.view(batch_size * seq_len, seq_len)  # (N, seq_len)

           # Create a distance mask: only keep neighbors with distance < threshold
            neighbor_mask = (distance_flat < distance_threshold).float().unsqueeze(-1)  # (N, seq_len, 1)

            # Ensure edge_proj_for_pooling has shape (N, seq_len, 128)
            if edge_proj_for_pooling.shape[1] == seq_len:

                masked_edge_proj = edge_proj_for_pooling * neighbor_mask  # (N, seq_len, 128)

                # Compute the number of neighbors for each node (avoid division by zero)
                neighbor_count = neighbor_mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # (N, 1, 1)

               # Masked average pooling (only aggregate edge features of neighbor nodes)
                edge_pooled_mean = masked_edge_proj.sum(dim=1) / neighbor_count.squeeze(-1)  # (N, 128)

               # Masked max pooling
                masked_edge_proj_for_max = masked_edge_proj + (1 - neighbor_mask) * (-1e9) 
                edge_pooled_max, _ = torch.max(masked_edge_proj_for_max, dim=1)  # (N, 128)
            else:
                # Shape mismatch, fall back to original pooling
                print(f"Warning: edge_proj shape mismatch. Expected seq_len={seq_len}, got {edge_proj_for_pooling.shape[1]}. Using pooling.")
                edge_pooled_mean = torch.mean(edge_proj_for_pooling, dim=1)
                edge_pooled_max, _ = torch.max(edge_proj_for_pooling, dim=1)
        else:
            # No distance matrix provided, use original global pooling
            edge_pooled_mean = torch.mean(edge_proj_for_pooling, dim=1)  # (N, 128)
            edge_pooled_max, _ = torch.max(edge_proj_for_pooling, dim=1)  # (N, 128)

        edge_pooled = (edge_pooled_mean + edge_pooled_max) / 2  # (N, 128)
        
        # 3. Cross-modal attention mechanism
        # Reshape features to (seq_len, batch_size, features) to fit MultiheadAttention
        node_proj_reshaped = node_proj.view(batch_size, seq_len, 128).transpose(0, 1)  # (seq_len, batch_size, 128)
        edge_pooled_reshaped = edge_pooled.view(batch_size, seq_len, 128).transpose(0, 1)  # (seq_len, batch_size, 128)
        
        attended_node, _ = self.cross_modal_attention(
            node_proj_reshaped, edge_pooled_reshaped, edge_pooled_reshaped
        )  # (seq_len, batch_size, 128)
        
        # Edge features as query, node features as key and value
        attended_edge, _ = self.cross_modal_attention(
            edge_pooled_reshaped, node_proj_reshaped, node_proj_reshaped
        )  # (seq_len, batch_size, 128)
        
        # Reshape back
        attended_node = attended_node.transpose(0, 1).contiguous().view(-1, 128)  # (N, 128)
        attended_edge = attended_edge.transpose(0, 1).contiguous().view(-1, 128)  # (N, 128)
        
        # 4. Feature fusion
        fused_input = torch.cat([attended_node, attended_edge], dim=-1)  # (N, 256)
        fused_features = self.modal_fusion(fused_input)  # (N, 128)
        
        # 5. Compute contrastive loss
        contrastive_loss = self.contrastive_loss(node_proj, edge_pooled)
        
        return fused_features, contrastive_loss

    def forward(self,device,seqs, use_GNN, distance_threshold, phipsi, DSSP, movement, quate, distance, mask, tokenizer, pretrained_lm, use_lm, return_contrastive_loss=True):

        # DSSP 
        emd_DSSP = self.embedding_DSSP(DSSP)  # (batch_size, sequence_length, 3)

        # PhiPsi 
        sin_phipsi = torch.sin(phipsi)  # (batch_size, sequence_length, 2)
        cos_phipsi = torch.cos(phipsi)  # (batch_size, sequence_length, 2)

        # lm
        lm_embedding = get_lm_embedding_(seqs, tokenizer, pretrained_lm, use_lm)

        node_feat_0 = torch.cat([emd_DSSP, sin_phipsi, cos_phipsi, lm_embedding], dim=-1)
        node_feat_0 = self.preprocess(node_feat_0)

        node_feat, _ = self.node_attention_layer(node_feat_0, node_feat_0, node_feat_0)
        node_feat = self.fc_norm1(node_feat + node_feat_0)  # res
        node_feat = self.fc_norm2(node_feat+self.fc(node_feat))


        node_feat = node_feat.reshape(-1, self.embed_dim)

        # edge
        distance_ = distance
        distance = distance.unsqueeze(-1)
        #print("movement shape:", movement.shape)
        #print("quate shape:", quate.shape)
        #print("distance shape:", distance.shape)
        edge_feat_0 = torch.cat([movement, quate, distance], dim=-1)
        #("edge_feat shape:", edge_feat.shape)
        #EFUN = CostRegNet().to(device)
        #edge_feat = EFUN(edge_feat_0)
        edge_feat = self.efun(edge_feat_0)

        edge_feat = self.edge_fc_norm1(edge_feat + edge_feat_0)
        edge_feat = self.edge_fc_norm2(edge_feat + self.edge_fc(edge_feat))

        #graphs,edge_feat = build_graph_from_distance_matrix(use_GNN,distance_threshold,distance_,edge_feat)

        batch_size = seqs.size(0)
        seq_len = seqs.size(1)

        fused_features, contrastive_loss = self.multimodal_alignment(
            node_feat, edge_feat, batch_size, seq_len,
            distance_matrix=distance_,  
            distance_threshold=distance_threshold 
        )
        
        # Reshape back to sequence form for sequence-level information aggregation
        fused_features_seq = fused_features.view(batch_size, seq_len, self.embed_dim)

        output = fused_features_seq
        preds = self.classifier(output)
        preds = preds.mean(dim=1)  #[64, 1]
        preds = preds.squeeze(-1)
        #print(f"preds shape after classifier: {preds.shape}")

        if return_contrastive_loss:
            return preds, contrastive_loss
        else:
            return preds
        