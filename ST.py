import torch
from torch import nn

class LayerNorm(torch.nn.Module):

    def __init__(self, dim, eps = 1e-6):
        super.__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):


class Attention():

    def __init__(self, heads, d_model):

    
    def forward(self, x):



class TransBlock():

    def __init__(self, d_model, heads, inter_d):
        self.Attention = Attention(heads, d_model)
        self.FNN = nn.Sequential(nn.Linear(d_model, inter_d),
                                 nn.ReLU(),
                                 nn.Linear(inter_d, d_model))
        
    def forward(self, x):
        x = self.Attention(x)
        x = LayerNorm(x) + x

class Encoder():

    def __init__(self, d_model, intput_dim, output_dim, vocab_size, N, heads, inter_d):
        self.PosEmbedding = nn.Linear(vocab_size, d_model)
        self.Transformer = nn.Sequential()
        for _ in range(N):
            self.Transformer.add_module(TransBlock(d_model, heads, inter_d))
        self.output = nn.Linear(d_model, vocab_size)

    
    def forward(self, x):
        x = self.PosEmbedding(x)
        x = self.Transformer(x)
