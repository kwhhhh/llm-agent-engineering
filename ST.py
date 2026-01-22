import torch
from torch import nn

class RMSNorm(nn.Module):

    def __init__(self, dim, eps = 1e-6):
        super.__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x):
        return x * (torch.rsqrt(x.pow(2).mean(-1, keepdim=True)) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output *= (self.weight.float())
        return output.type_as(x)

class MHAttention(nn.Module):

    def __init__(self, heads, d_model):
        super(MHAttention).__init__()
        assert d_model % heads == 0
        self.d_model = d_model
        self.heads = heads
        self.d_k = d_model // heads
        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
    
    def scale_dot_product_attention(self, Q, K, V, mask = None):
        attn_score = torch.matmul(Q, K.transpose(-2,-1)) / torch.sqrt(self.d_k)

        if mask is None:
            attn_score = attn_score
        else:
            attn_score = attn_score.masked_fill(mask=0, -1e9)

        attn_probs = torch.softmax(attn_score, dim = -1)
        score = torch.matmul(attn_probs, V)

        return score

    def forward(self, x):



class TransBlock():

    def __init__(self, d_model, heads, inter_d):
        self.Attention = Attention(heads, d_model)
        self.FNN = nn.Sequential(nn.Linear(d_model, inter_d),
                                 nn.ReLU(),
                                 nn.Linear(inter_d, d_model))
        
    def forward(self, x):
        x = self.Attention(x)
        x = RMSNorm(x) + x

class Encoder():

    def __init__(self, d_model, intput_dim, output_dim, vocab_size, N, heads, inter_d, dropout):
        self.PosEmbedding = nn.Linear(vocab_size, d_model)
        self.Transformer = nn.Sequential()
        for _ in range(N):
            self.Transformer.add_module(TransBlock(d_model, heads, inter_d))
        self.dropout = dropout
        self.output = nn.Linear(d_model, vocab_size)

    
    def forward(self, x):
        x = self.PosEmbedding(x)
        x = self.Transformer(x)
