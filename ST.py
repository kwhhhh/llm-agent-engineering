import torch
from torch import nn
import torch.nn.functional as F

class RMSNorm(nn.Module):

    def __init__(self, dim, eps = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x):
        return x * (torch.rsqrt(x.pow(2).mean(-1, keepdim=True)) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output *= (self.weight.float())
        return output.type_as(x)

class MHAttention(nn.Module):

    def __init__(self, heads, d_model, dropout):
        super().__init__()
        assert d_model % heads == 0
        self.d_model = d_model
        self.heads = heads
        self.d_k = d_model // heads
        # self.dropout = dropout
        # self.resid_dropout = nn.Dropout(self.dropout)
        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
    
    # def scale_dot_product_attention(self, Q, K, V, mask = None):
    #     attn_score = torch.matmul(Q, K.transpose(-2,-1)) / torch.sqrt(self.d_k)

    #     if mask is None:
    #         attn_score = attn_score
    #     else:
    #         attn_score = attn_score.masked_fill(mask==0, -1e9)

    #     attn_probs = torch.softmax(attn_score, dim = -1)
    #     score = torch.matmul(attn_probs, V)

    #     return score

    def forward(self, x):
        bsz, seq_len, _ = x.shape
        xk, xq, xv = self.WK(x), self.WQ(x), self.WV(x)
        xk = xk.view(bsz, seq_len, self.heads, self.d_k).transpose(1, 2)
        xq = xq.view(bsz, seq_len, self.heads, self.d_k).transpose(1, 2)
        xv = xv.view(bsz, seq_len, self.heads, self.d_k).transpose(1, 2)

        output = F.scaled_dot_product_attention(xk, xq, xv, dropout_p=0.01)
        attn_out = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)  # (bsz, seq_len, d_model)

        # output.transpose(1,2).reshape(bsz, seq_len, -1)
        # output = self.Wo(self.resid_dropout(output))
        output = self.Wo(attn_out)
        return output

class TransBlock(nn.Module):

    def __init__(self, d_model, heads, inter_d, dropout=0.01):
        super().__init__()
        self.Attention = MHAttention(heads, d_model,dropout)
        self.FNN = nn.Sequential(nn.Linear(d_model, inter_d),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(inter_d, d_model))
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        attn_out = self.Attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        FFN_output = self.FNN(x)
        x = self.norm2(x + self.dropout(FFN_output))

        return x

        # pass

class Encoder(nn.Module):

    def __init__(self, d_model, intput_dim, output_dim, vocab_size, N, heads, inter_d, dropout):
        super().__init__()
        self.PosEmbedding = nn.Linear(vocab_size, d_model)
        # self.Transformer = nn.Sequential()
        self.Transformer = nn.Sequential(
            TransBlock(d_model, heads, inter_d),
            TransBlock(d_model, heads, inter_d),
            TransBlock(d_model, heads, inter_d)
        )
        # nn.Sequential()
        # for _ in range(N):
        #     self.Transformer.add_module(TransBlock(d_model, heads, inter_d))
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(d_model, vocab_size)

    
    def forward(self, x):
        x = self.PosEmbedding(x)
        x = self.Transformer(x)

        return self.output(self.dropout(x))

if __name__ == "__main__":
    Attn = Encoder(1000000, 30, 100, 100, 3, 5, 16, 0.01)
    x = torch.arange(1200, dtype=torch.float32).reshape(3,4,100)
    print(x)
    output = Attn(x)
    print(output)
    print(output.shape)