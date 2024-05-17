import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias
        x = torch.softmax(x, dim=3)
        x = torch.where(torch.isnan(x), torch.full_like(x, 0), x)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class Transformer(nn.Module):
    def __init__(self, in_dim, pe_dim, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(Transformer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)
        self.node_add = nn.Linear(in_dim, hidden_size - pe_dim)

    def forward(self, x, pe, coarse_batch, mask=None, attn_bias=None):
        if mask is None:
            mask = torch.ones(x.size(0), dtype=torch.long, device=x.device)
        keep_nodes = torch.where(mask == 1)[0]
        mask_nodes = torch.where(mask == 0)[0]
        final_x = torch.zeros_like(x, device=x.device)
        keep_x = x[keep_nodes]
        keep_pe = pe[keep_nodes]
        keep_batch = coarse_batch[keep_nodes]
        mask_x = x[mask_nodes]
        keep_x = self.node_add(keep_x)
        keep_x = torch.cat([keep_x, keep_pe], dim=-1)
        keep_x, mask = to_dense_batch(keep_x, keep_batch)
        y = self.self_attention_norm(keep_x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        keep_x = keep_x + y
        keep_x = keep_x[mask]
        y = self.ffn_norm(keep_x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        keep_x = keep_x + y
        final_x[keep_nodes] = keep_x
        final_x[mask_nodes] = mask_x
        return final_x