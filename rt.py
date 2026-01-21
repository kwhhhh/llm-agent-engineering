import torch
from torch import nn


class Transformer():

    def __init__(self, d_model, ):
        self.WQ = nn.Linear()
        self.WK = nn.Linear()
        self.WV = nn.Linear()
        

class Encoder():

    def __init__(self, input_dim, d_model, heads, N):
        self.FFN = FFNBlock() # 待实现
        # self.Attention = Attention(d_model, d_model, heads) # 注意力机制
        self.PosEmbedding = PosEmbedding(input_dim, d_model)
        self.Blocks = nn.Sequential(Transformer() for _ in range(N))

    
    def forward(self, x):
        x = self.PosEmbedding(x)
