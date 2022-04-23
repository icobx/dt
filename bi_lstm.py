import torch
import torch.nn as nn
import numpy as np

from bert_embedding_model import BertEmbeddingModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Attention(nn.Module):
    """
    Attention module inspired by:
    - https://discuss.pytorch.org/t/self-attention-on-words-and-masking/5671
    - https://www.kaggle.com/code/dannykliu/lstm-with-attention-clr-in-pytorch/notebook
    """

    def __init__(self, hidden_dim, device):
        super(Attention, self).__init__()

        self.hidden_dim = hidden_dim
        self.device = device
        self.attention_weights = nn.Parameter(torch.Tensor(1, hidden_dim), requires_grad=True).to(device)
        self.relu = nn.ReLU()

        # stdv = 1.0 / np.sqrt(self.hidden_dim)
        # for weight in self.attention_weights:
        #     nn.init.uniform_(weight, -stdv, stdv)
        nn.init.xavier_uniform_(self.attention_weights)

    def get_mask(self):
        pass

    def forward(self, inputs, lengths):
        batch_size, max_len = inputs.size()[:2]

        # swap dimensions (permute), add unitary dimension at the start (unsqueeze), repeat the the tensor (repeaat)
        weights = self.attention_weights.permute(1, 0).unsqueeze(0).repeat(batch_size, 1, 1)
        # print('weights pre bmm: ', weights.size())
        # print('inputs pre bmm: ', inputs.size())

        # print('pre bmm: ', weights)
        # apply attention layer
        weights = torch.bmm(
            inputs,
            weights
        )
        # print('weights post bmm', weights.size())
        # print('weights squeezed', weights.squeeze().size())
        # print('weights squeezed', weights.squeeze())
        attentions = torch.softmax(self.relu(weights.squeeze()), dim=-1)
        # print('attentions', attentions.size())
        # print('attentions', attentions)

        mask = torch.ones(attentions.size(), requires_grad=True).to(self.device)
        with torch.no_grad():
            for i, l in enumerate(lengths):
                if l < max_len:
                    mask[i, l:] = 0

        masked = attentions * mask
        # print('masked', masked.size())
        # print('masked', masked)
        _sums = masked.sum(-1).unsqueeze(-1)  # sums per row
        # print('_sums', _sums.size())
        # print('_sums', _sums)
        attentions = masked.div(_sums)
        # print('attentions after masked:', attentions.size())
        # print('attentions after masked:', attentions)
        # apply attention weights
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))
        # print('weighted', weighted.size())
        # print('weighted', weighted)
        # get the final fixed vector representations of the sentences
        # representations = weighted.sum(1).squeeze()
        # print('representations', representations.size())
        # print('representations', representations)
        return weighted, attentions


class BiLSTM(nn.Module):

    def __init__(
        self,
        dropout=0.5,
        lstm_dropout=0,
        hidden_dim=128,
        embedding_dim=768,
        sent_level_feature_dim=0,
        device=torch.device('cpu'),
        w_seq=False
    ):
        super(BiLSTM, self).__init__()

        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout
        )
        self.attention = Attention(hidden_dim=2*hidden_dim, device=device)
        self.drop = nn.Dropout(p=dropout)
        self.seq = None
        if w_seq:
            self.seq = nn.Sequential(
                nn.Linear(2*hidden_dim+sent_level_feature_dim, 2*hidden_dim+sent_level_feature_dim),
                nn.BatchNorm1d(2*hidden_dim+sent_level_feature_dim),
                #                 nn.ReLU()
            )
        self.dense = nn.Linear(2*hidden_dim+sent_level_feature_dim, 1)

    def forward(self, embeddings, lengths, sent_level_features=None):
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)

        output, lens = pad_packed_sequence(packed_out, batch_first=True)
        output, _ = self.attention(output, lens)

        out_forward = output[range(len(output)), lengths - 1, :self.hidden_dim]
        out_reverse = output[:, 0, self.hidden_dim:]

        out = torch.cat((out_forward, out_reverse), 1)
        out = self._append_features(out, sent_level_features, cat_dim=1)

        if self.seq:
            out = self.drop(out)
            out = self.seq(out)

        out = self.drop(out)
#         print('pre dense out size: ', out.size())
        out = self.dense(out)

        return torch.squeeze(out, 1)

    @staticmethod
    def _append_features(x, features, cat_dim):
        if features is None or features.size()[-1] == 0:
            return x
     
        return torch.cat((x, features), dim=cat_dim).float()

    @staticmethod
    def _pad(emb_tensors):
        lens = [len(s) for s in emb_tensors]
        max_len = max(lens)

        matrix = np.zeros((len(emb_tensors), max_len, emb_tensors[0].shape[1]))

        for i in range(len(emb_tensors)):
            matrix[i, :lens[i], :] = emb_tensors[i]

        return torch.FloatTensor(matrix), lens


# bem = BertEmbeddingModel()

# xx, xxx = bem(['You hired some workers from Poland...', 'This is a testing sentence.'])

# bi = BiLSTM()

# yy = bi(xx, xxx)

# print(yy.size())
# print(torch.sigmoid(yy))
