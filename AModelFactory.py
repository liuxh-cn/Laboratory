import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

class ModelFactory():

    @staticmethod
    def createTransformer(Dict, emsize, nhead, nhid, nlayers, dropout=0.5):
        return TransformerModel(Dict, emsize, nhead, nhid, nlayers, dropout)

class PositionalEncoding(nn.Module):
    '''
    Input:  [35, 20, 200] ← word_embedding
    Output: [35, 20, 200] ← word_embedding + pos_embedding
    '''
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
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):

    def __init__(self, Dict, emsize, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()

        # In - Embedding Dimension
        self.src_mask = None
        self.emsize = emsize
        self.Dict = Dict

        # Model - Position Embedding & Transformer Encoder
        self.pos_encoder = PositionalEncoding(emsize, dropout)
        encoder_layers = TransformerEncoderLayer(emsize, nhead, nhid, dropout)    # emsize, nhead, nhiden
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # Out - Fully Connected Layer: In-emsize; Out-vacsize
        self.decoder = nn.Linear(emsize, 1)

        # In & Out <= Weights Initialization
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        
        r = torch.tensor([list(map(lambda x: self.Dict.loc[int(x)].values, i)) for i in src], dtype=torch.float32)
        src = r * math.sqrt(self.emsize)  # ninp = embedding dimension
        # word_embedding + position_embedding           [35, 20, 200]
        src = self.pos_encoder(src)

        output = self.transformer_encoder(src, self.src_mask)
        # output                                        [35, 20, 200] => [35, 20, 28708]
        output = self.decoder(output)
        return output

if __name__ == '__main__':
    p = PositionalEncoding(5, 0.1)
