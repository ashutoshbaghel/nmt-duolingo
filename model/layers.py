import math

import torch
import torch.nn.functional as F
from torch import nn


def attention(q, k, v, d_k, mask, dropout):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    scores = dropout(F.softmax(scores.masked_fill(mask == 0, -1e9), dim=-1))

    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.d_model = opt.emb_dim
        self.d_k = int(self.d_model / opt.heads)
        self.h = opt.heads

        self.q_linear = nn.Linear(self.d_model, self.d_model)
        self.v_linear = nn.Linear(self.d_model, self.d_model)
        self.k_linear = nn.Linear(self.d_model, self.d_model)

        self.dropout = nn.Dropout(opt.dropout)
        self.out = nn.Linear(self.d_model, self.d_model)

    def forward(self, q, k, v, mask):
        k = self.k_linear(k).view(q.size(0), -1, self.h, self.d_k).transpose(1, 2)
        q = self.q_linear(q).view(q.size(0), -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(q.size(0), -1, self.h, self.d_k).transpose(1, 2)

        mask = mask.unsqueeze(1)
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        concat = scores.transpose(1, 2).contiguous() \
            .view(q.size(0), -1, self.d_model)
        output = self.out(concat)

        return output


class EncoderLayer(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.norm_1 = Norm(opt.emb_dim)
        self.norm_2 = Norm(opt.emb_dim)

        self.dropout_1 = nn.Dropout(opt.dropout)
        self.dropout_2 = nn.Dropout(opt.dropout)

        self.attn = MultiHeadAttention(opt)

        self.ff = FeedForward(opt)

        self.d = math.sqrt(opt.emb_dim // opt.heads)

    def forward(self, x, mask):
        '''
        This implementation follows the Tensor2Tensor implementation
        instead of the original paper "Attention is all you need"
        The Norm is applied to the input first, then self attention
        is applied to the sub-layer.
        '''

        x = self.norm_1(x)
        x1 = x + self.dropout_1(self.attn(x, x, x, mask))

        x1 = self.norm_2(x1)
        x2 = x1 + self.dropout_2(self.ff(x1))

        return x2


class DecoderLayer(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.norm_1 = Norm(opt.emb_dim)
        self.norm_2 = Norm(opt.emb_dim)
        self.norm_3 = Norm(opt.emb_dim)

        self.dropout_1 = nn.Dropout(opt.dropout)
        self.dropout_2 = nn.Dropout(opt.dropout)
        self.dropout_3 = nn.Dropout(opt.dropout)

        self.attn_1 = MultiHeadAttention(opt)

        self.attn_2 = MultiHeadAttention(opt)

        self.ff = FeedForward(opt)

        self.d = math.sqrt(opt.emb_dim // opt.heads)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        '''
        This implementation follows the Tensor2Tensor implementation
        instead of the original paper "Attention is all you need"
        The Norm is applied to the input first, then self attention
        is applied to the sub-layer.
        '''
        x = self.norm_1(x)
        x1 = x + self.dropout_1(self.attn_1(x, x, x, trg_mask))

        x1 = self.norm_2(x1)
        x2 = x1 + self.dropout_2(self.attn_2(x1,
                                             e_outputs,
                                             e_outputs,
                                             src_mask))

        x2 = self.norm_3(x2)
        x3 = x2 + self.dropout_3(self.ff(x2))

        return x3


class PositionalEncoder(nn.Module):
    def __init__(self, opt, max_seq_len=100):
        super().__init__()
        self.dropout = nn.Dropout(p=opt.dropout)
        self.dim = opt.emb_dim
        pe = torch.zeros(max_seq_len, self.dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # 1000 ^ (2i / dmodel) = e ^ (2i) * -log(1000)
        div_term = torch.exp(torch.arange(0, self.dim, 2).float() \
                             * (-math.log(10000.0) / self.dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.dim)
        pe = Variable(self.pe[:, :x.size(1)], requires_grad=False)
        pe = pe.to(device)
        x = x + pe
        return self.dropout(x)


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()

        self.size = d_model

        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class FeedForward(nn.Module):
    def __init__(self, opt):
        super().__init__()

        linear_1 = nn.Linear(opt.emb_dim, opt.ff_hsize)
        dropout = nn.Dropout(opt.dropout)
        linear_2 = nn.Linear(opt.ff_hsize, opt.emb_dim)

        self.layers = nn.Sequential(linear_1, nn.ReLU(), dropout, linear_2)

    def forward(self, x):
        self.layers(x)
        return x


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)
