import torch
import torch.nn as nn

from bert_embedding_model import BertEmbeddingModel
from feature_nn import FeatureNN
from bi_lstm import BiLSTM


class PairNN(nn.Module):

    def __init__(self, subnetwork_0, subnetwork_1, hidden_dim, dropout):
        super(PairNN, self).__init__()

        self.sn0 = subnetwork_0
        self.sn1 = subnetwork_1 if not subnetwork_1.is_unused else None

        self.dropout = nn.Dropout(p=dropout)
        self.seq = nn.Sequential(
            nn.Linear(1 if subnetwork_1.is_unused else 2, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, input0, input1):
        out = torch.unsqueeze(self.sn0(*input0), 1)
        
        if self.sn1:
            out_feat = torch.unsqueeze(self.sn1(input1), 1)
            out = torch.cat((out, out_feat), 1)
        
        return torch.squeeze(self.seq(out), 1)


# bem = BertEmbeddingModel()

# xx, xxx = bem(['You hired some workers from Poland...', 'This is a testing sentence.'])
# features = torch.randn((2, 64))
# bi = BiLSTM()
# fn = FeatureNN(64)

# pn = PairNN(bi, fn, 128, 0.1)

# yy = pn((xx, xxx), features)
# # yy = bi(xx, xxx)
