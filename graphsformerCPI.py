#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''==============================================
# @Project : DGraphDTA-master
# @File    : graphsformerCPI.py
# @IDE     : PyCharm
# @Author  : Austin
# @Time    : 2022/10/26 16:33
================================================'''
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class Embeddings(nn.Module):
    def __init__(self, d_feature, d_model, dropout):
        super(Embeddings, self).__init__()
        self.ln = nn.Linear(d_feature, d_model)
        self.activation_fun = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.dropout(self.activation_fun(self.ln(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len, d_model]
    seq_k: [batch_size, seq_len, d_model]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q, _ = seq_q.size()
    batch_size, len_k, _ = seq_k.size()

    pad_attn_mask = torch.sum(torch.abs(seq_k), dim=-1) == 0  # [bitch_size, len_k]
    pad_attn_mask = pad_attn_mask.unsqueeze(1)
    attn_mask = pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

    return attn_mask


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, enc_dec_flag, dropout=0.1, activation_fun='softmax'):
        super(ScaledDotProductAttention, self).__init__()
        self.enc_dec_flag = enc_dec_flag
        self.d_k = d_k
        if self.enc_dec_flag:
            self.conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)

        if activation_fun == 'softmax':
            self.activation_fun = nn.Softmax(dim=-1)
        elif activation_fun == 'relu':
            self.activation_fun = nn.ReLU()
        elif activation_fun == 'sigmoid':
            self.activation_fun = nn.Sigmoid()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, Q, K, V, attn_mask_col, adj_matrix=None, dist_matrix=None):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        adj_matrix,dist_matrix,attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        # scores: [batch_size, n_heads, len_q, len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores = scores.masked_fill(attn_mask_col, -1e9)
        scores = self.activation_fun(scores)

        if self.enc_dec_flag:  # true:encoder, false:decoder
            # 法1：1*1卷积
            batch_size, n_heads, len_q, len_k = scores.shape

            scores = scores.reshape(-1, len_q, len_k)
            dist_matrix = dist_matrix.reshape(-1, len_q, len_k)
            adj_matrix = adj_matrix.reshape(-1, len_q, len_k)

            con_matrix = torch.stack([scores, dist_matrix, adj_matrix], dim=1)

            weighted_sores = self.conv(con_matrix)
            weighted_sores = weighted_sores.squeeze(1)
            weighted_sores = weighted_sores.reshape(batch_size, n_heads, len_q, len_k)
            weighted_sores = weighted_sores.masked_fill(attn_mask_col, -1e9)
            weighted_sores = self.activation_fun(weighted_sores)

        else:
            weighted_sores = scores

        attn = self.dropout(weighted_sores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, d_v,d_model]

        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout, enc_dec_flag, distance_matrix_kernel='exp', activation_fun='softmax'):
        super(MultiHeadAttention, self).__init__()

        assert d_model % n_heads == 0
        self.enc_dec_flag = enc_dec_flag
        self.d_v = self.d_k = d_model // n_heads
        self.d_model = d_model
        self.n_heads = n_heads

        self.W_Q = nn.Linear(d_model, self.d_k * self.n_heads, bias=False)
        self.W_K = nn.Linear(d_model, self.d_k * self.n_heads, bias=False)
        self.W_V = nn.Linear(d_model, self.d_v * self.n_heads, bias=False)
        self.fc = nn.Linear(n_heads * self.d_v, self.d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout)

        if self.enc_dec_flag:  # true:encoder, false:decoder
            self.dot_product_atten = ScaledDotProductAttention(self.d_k, self.enc_dec_flag, dropout, activation_fun)

            if distance_matrix_kernel == 'softmax':
                self.distance_matrix_kernel = lambda x: F.softmax(-x, dim=-1)
            elif distance_matrix_kernel == 'exp':
                self.distance_matrix_kernel = lambda x: torch.exp(-x)
            elif distance_matrix_kernel == 'sigmoid':
                self.distance_matrix_kernel = lambda x: torch.sigmoid(-x)
        else:
            self.dot_product_atten = ScaledDotProductAttention(d_k=self.d_k, enc_dec_flag=self.enc_dec_flag,
                                                               dropout=dropout)

    def forward(self, input_Q, input_K, input_V, attn_mask, adj_matrix=None, dist_matrix=None):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual = input_Q
        batch_size = input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,
                                                                                     2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,
                                                                                     2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,
                                                                                     2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        # Prepare distances matrix，dist_matrix对应f(-x)函数

        if self.enc_dec_flag:
            dist_matrix = dist_matrix.masked_fill(attn_mask, np.inf)
            dist_matrix = self.distance_matrix_kernel(dist_matrix)  # 补加行没有被mask,在attension模块中与其他特征一起处理
            dist_matrix = dist_matrix.unsqueeze(1).repeat(1, self.n_heads, 1,
                                                          1)  # [batch_size, n_heads, seq_len, seq_len]

            adj_matrix = adj_matrix.unsqueeze(1).repeat(1, self.n_heads, 1,
                                                        1)  # [batch_size, n_heads, seq_len, seq_len]

            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1,
                                                      1)  # [batch_size, n_heads, seq_len, seq_len]

            context, attn = self.dot_product_atten(Q, K, V, attn_mask, adj_matrix, dist_matrix)
        else:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1,
                                                      1)  # [batch_size, n_heads, seq_len, seq_len]
            context, attn = self.dot_product_atten(Q, K, V, attn_mask)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                            self.n_heads * self.d_v)  # context: [batch_size, len_model, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_v, d_model]

        return output, attn  # ,同时，ffn的正则化提到开始，也做相同的残差连接。


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ffn, dropout):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = d_model
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ffn, bias=False),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_ffn, d_model, bias=False)
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        output = self.fc(inputs)
        return output  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ffn, n_heads, dropout, distance_matrix_kernel, activation_fun, enc_dec_flag):
        super(EncoderLayer, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNorm(d_model)
        self.enc_self_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout,
                                                enc_dec_flag=enc_dec_flag,
                                                distance_matrix_kernel=distance_matrix_kernel,
                                                activation_fun=activation_fun)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ffn, dropout)

    def forward(self, enc_inputs, adj_matrix, dist_matrix, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        residual = enc_inputs
        enc_inputs = self.norm(enc_inputs)
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask, adj_matrix,
                                               dist_matrix)  # enc_inputs to same Q,K,V
        residual = residual + self.dropout(enc_outputs)
        ffn_inputs = self.norm(residual)
        ffn_outputs = self.pos_ffn(ffn_inputs)

        return residual + self.dropout(ffn_outputs), attn


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout, d_ffn, distance_matrix_kernel, activation_fun, enc_dec_flag):
        super(DecoderLayer, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNorm(d_model)
        # 调用两次 MultiHeadAttention
        self.cpd_prt_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout,
                                               enc_dec_flag=enc_dec_flag,
                                               distance_matrix_kernel=distance_matrix_kernel,
                                               activation_fun=activation_fun)
        self.cpd_prt_ffn = PoswiseFeedForwardNet(d_model=d_model, d_ffn=d_ffn, dropout=dropout)

        self.prt_cpd_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout,
                                               enc_dec_flag=enc_dec_flag,
                                               distance_matrix_kernel=distance_matrix_kernel,
                                               activation_fun=activation_fun)
        self.prt_cpd_ffn = PoswiseFeedForwardNet(d_model=d_model, d_ffn=d_ffn, dropout=dropout)

    def forward(self, cpd_enc_output, prt_enc_output, cpd_prt_attn_mask, prt_cpd_attn_mask):
        cpd_residual = cpd_enc_output
        prt_residual = prt_enc_output
        cpd_enc_output = self.norm(cpd_enc_output)
        prt_enc_output = self.norm(prt_enc_output)
        dec_cpd_outputs, cpd_prt_attn = self.cpd_prt_attn(cpd_enc_output, prt_enc_output, prt_enc_output,
                                                          cpd_prt_attn_mask)  # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_prt_outputs, prt_cpd_attn = self.prt_cpd_attn(prt_enc_output, cpd_enc_output, cpd_enc_output,
                                                          prt_cpd_attn_mask)

        cpd_residual = cpd_residual + self.dropout(dec_cpd_outputs)
        prt_residual = prt_residual + self.dropout(dec_prt_outputs)

        ffn_cpd_inputs = self.norm(cpd_residual)
        ffn_prt_inputs = self.norm(prt_residual)

        ffn_cpd_outputs = self.cpd_prt_ffn(ffn_cpd_inputs)  # [batch_size, tgt_len, d_model]
        ffn_prt_outputs = self.prt_cpd_ffn(ffn_prt_inputs)

        ffn_cpd_outputs = cpd_residual + self.dropout(ffn_cpd_outputs)
        ffn_prt_outputs = prt_residual + self.dropout(ffn_prt_outputs)

        return ffn_cpd_outputs, ffn_prt_outputs, cpd_prt_attn, prt_cpd_attn


class Enc_Dec_Layer(nn.Module):
    def __init__(self, d_model=128, n_heads=8, dropout=0.1, d_ffn=256, distance_matrix_kernel='softmax',
                 activation_fun='softmax'):
        super(Enc_Dec_Layer, self).__init__()

        self.cpd_enc_layer = EncoderLayer(d_model, d_ffn, n_heads, dropout, distance_matrix_kernel, activation_fun,
                                          enc_dec_flag=True)
        self.prt_enc_layer = EncoderLayer(d_model, d_ffn, n_heads, dropout, distance_matrix_kernel, activation_fun,
                                          enc_dec_flag=True)
        self.cpd_prt_dec_layer = DecoderLayer(d_model, n_heads, dropout, d_ffn, distance_matrix_kernel, activation_fun,
                                              enc_dec_flag=False)

        self.layer_norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, cpd_atom_features, cpd_adj_matrix, cpd_dist_matrix, cpd_self_attn_mask, prt_aa_features,
                prt_contact_map, prt_dist_matrix, prt_self_attn_mask, cpd_prt_attn_mask, prt_cpd_attn_mask):
        cpd_enc_output, cpd_enc_attn = self.cpd_enc_layer(cpd_atom_features, cpd_adj_matrix, cpd_dist_matrix,
                                                          cpd_self_attn_mask)
        prt_enc_output, prt_enc_attn = self.prt_enc_layer(prt_aa_features, prt_contact_map, prt_dist_matrix,
                                                          prt_self_attn_mask)

        cpd_output, prt_output, cpd_prt_attn, prt_cpd_attn = self.cpd_prt_dec_layer(cpd_enc_output,
                                                                                    prt_enc_output,
                                                                                    cpd_prt_attn_mask,
                                                                                    prt_cpd_attn_mask)

        cpd_output = cpd_enc_output + self.dropout(cpd_output)
        prt_output = prt_enc_output + self.dropout(prt_output)

        return cpd_output, prt_output, cpd_enc_attn, prt_enc_attn, cpd_prt_attn, prt_cpd_attn


class Encoder_Decoder(nn.Module):
    def __init__(self, cpd_atom, prt_aa, layers=2, d_model=128, n_heads=8, dropout=0.1,
                 distance_matrix_kernel='softmax',
                 d_ffn=256, activation_fun='softmax'):
        super(Encoder_Decoder, self).__init__()

        self.comp_emb = Embeddings(cpd_atom, d_model, dropout)  # 这个是去定义生成一个矩阵，大小是 src_vocab_size * d_model
        self.comp_pos_emb = PositionalEncoding(d_model)

        self.prt_emb = Embeddings(prt_aa, d_model, dropout)  # 这个是去定义生成一个矩阵，大小是 src_vocab_size * d_model
        self.prt_pos_emb = PositionalEncoding(d_model)

        self.layer_norm = LayerNorm(d_model)

        self.layers = nn.ModuleList(
            [Enc_Dec_Layer(d_model=d_model, n_heads=n_heads, dropout=dropout, d_ffn=d_ffn,
                           distance_matrix_kernel=distance_matrix_kernel, activation_fun=activation_fun) for _ in
             range(layers)])

    def forward(self, cpd_atom_features, cpd_adj_matrix, cpd_dist_matrix, cpd_self_attn_mask, prt_aa_features,
                prt_contact_map, prt_dist_matrix, prt_self_attn_mask, cpd_prt_attn_mask, prt_cpd_attn_mask):
        cpd_enc_output = self.comp_emb(cpd_atom_features)  # [batch_size, src_len, d_model]
        cpd_enc_output = self.comp_pos_emb(cpd_enc_output.transpose(0, 1)).transpose(0, 1)

        prt_enc_output = self.prt_emb(prt_aa_features)  # [batch_size, src_len, d_model]
        prt_enc_output = self.prt_pos_emb(prt_enc_output.transpose(0, 1)).transpose(0, 1)

        cpd_enc_attn_list, prt_enc_attn_list = [], []
        cpd_prt_attn_list = []
        for layer in self.layers:
            cpd_enc_output, prt_enc_output, cpd_enc_attn, prt_enc_attn, cpd_prt_attn, _ = layer(
                cpd_atom_features=cpd_enc_output, cpd_adj_matrix=cpd_adj_matrix, cpd_dist_matrix=cpd_dist_matrix,
                cpd_self_attn_mask=cpd_self_attn_mask, prt_aa_features=prt_enc_output,
                prt_contact_map=prt_contact_map, prt_dist_matrix=prt_dist_matrix, prt_self_attn_mask=prt_self_attn_mask,
                cpd_prt_attn_mask=cpd_prt_attn_mask, prt_cpd_attn_mask=prt_cpd_attn_mask)

            cpd_enc_attn_list.append(cpd_enc_attn)
            prt_enc_attn_list.append(prt_enc_attn)
            cpd_prt_attn_list.append(cpd_prt_attn)

        cpd_enc_output = self.layer_norm(cpd_enc_output)
        prt_enc_output = self.layer_norm(prt_enc_output)
        return cpd_enc_output, prt_enc_output, cpd_enc_attn_list, prt_enc_attn_list, cpd_prt_attn_list


class Predictor(nn.Module):
    def __init__(self, d_model, dropout=0.1, n_output=1):
        super(Predictor, self).__init__()

        self.cpd_dropout = nn.Dropout(p=dropout)
        self.prt_dropout = nn.Dropout(p=dropout)

        self.W1 = nn.Linear(d_model, 1, bias=False)
        self.W2 = nn.Linear(d_model, 1, bias=False)

        self.activation_fun = nn.ReLU()

        self.softmax1 = nn.Softmax(dim=-1)
        self.softmax2 = nn.Softmax(dim=-1)

        self.ln = nn.Linear(2 * d_model, n_output, bias=False)

    def forward(self, cpd_atom_feats, prt_aa_feats):
        batch_size = cpd_atom_feats.size(0)

        W1_output = self.cpd_dropout(self.activation_fun(self.W1(cpd_atom_feats)))
        W1_output = W1_output.view(batch_size, -1)
        W1_output = self.softmax1(W1_output)

        W2_output = self.prt_dropout(self.activation_fun(self.W2(prt_aa_feats)))
        W2_output = W2_output.view(batch_size, -1)
        W2_output = self.softmax2(W2_output)

        cf = torch.sum(cpd_atom_feats * W1_output.view(batch_size, -1, 1), dim=1)
        pf = torch.sum(prt_aa_feats * W2_output.view(batch_size, -1, 1), dim=1)
        cat_cp = torch.cat((cf, pf), dim=-1)

        return self.ln(cat_cp)


class graphsformerCPI(nn.Module):
    def __init__(self, cpd_atom, prt_aa, layers=2, d_model=128, n_heads=8, dropout=0.1,
                 distance_matrix_kernel='softmax',
                 d_ffn=256, n_output=1, activation_fun='softmax'):
        super(graphsformerCPI, self).__init__()

        self.enc_dec = Encoder_Decoder(cpd_atom, prt_aa, layers, d_model, n_heads, dropout, distance_matrix_kernel,
                                       d_ffn, activation_fun)

        self.predictor = Predictor(d_model, dropout, n_output)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.3)
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0.1)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zeros_()

    def forward(self, cpd_atom_features, cpd_adj_matrix, cpd_dist_matrix, prt_aa_features,
                prt_contact_map, prt_dist_matrix):

        cpd_self_attn_mask = get_attn_pad_mask(cpd_atom_features, cpd_atom_features)
        prt_self_attn_mask = get_attn_pad_mask(prt_aa_features, prt_aa_features)

        cpd_prt_attn_mask = get_attn_pad_mask(cpd_atom_features, prt_aa_features)  # [batch_size, comp_len, prt_len]
        prt_cpd_attn_mask = get_attn_pad_mask(prt_aa_features, cpd_atom_features)  # [batch_size,prt_len, comp_len]

        cpd_enc_output, prt_enc_output, cpd_enc_attn_list, prt_enc_attn_list, cpd_prt_attn_list = self.enc_dec(
            cpd_atom_features=cpd_atom_features, cpd_adj_matrix=cpd_adj_matrix, cpd_dist_matrix=cpd_dist_matrix,
            cpd_self_attn_mask=cpd_self_attn_mask, prt_aa_features=prt_aa_features, prt_contact_map=prt_contact_map,
            prt_dist_matrix=prt_dist_matrix, prt_self_attn_mask=prt_self_attn_mask,
            cpd_prt_attn_mask=cpd_prt_attn_mask, prt_cpd_attn_mask=prt_cpd_attn_mask)

        cpd_mask = torch.sum(torch.abs(cpd_atom_features), dim=-1) == 0
        cpd_mask = cpd_mask.unsqueeze(-1).float()

        prt_mask = torch.sum(torch.abs(prt_aa_features), dim=-1) == 0
        prt_mask = prt_mask.unsqueeze(-1).float()

        logits = self.predictor(cpd_enc_output, prt_enc_output, cpd_mask, prt_mask)

        return logits, cpd_enc_attn_list, prt_enc_attn_list, cpd_prt_attn_list