from bert_embedding_model import BertEmbeddingModel
import torch
import torch.nn as nn
import numpy as np

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTM(nn.Module):

    def __init__(
        self,
        dropout=0.5,
        lstm_dropout=0,
        hidden_dim=128,
        embedding_dim=768,
        sent_level_feature_dim=0
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
        self.drop = nn.Dropout(p=dropout)
        self.dense = nn.Linear(2*hidden_dim+sent_level_feature_dim, 1)

    def forward(self, embeddings, lengths, sent_level_features=None):
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(packed_out, batch_first=True)

        out_forward = output[range(len(output)), lengths - 1, :self.hidden_dim]
        out_reverse = output[:, 0, self.hidden_dim:]

        out = torch.cat((out_forward, out_reverse), 1)
        # print(out)
        out = self._append_features(out, None, cat_dim=1)

        out_drop = self.drop(out)
        out_dense = self.dense(out_drop)

        return torch.squeeze(out_dense, 1)

    @staticmethod
    def _append_features(x, features, cat_dim):
        if features is None or len(features) == 0:
            return x

        return torch.cat((x, features), dim=cat_dim)

    @staticmethod
    def _pad(emb_tensors):
        lens = [len(s) for s in emb_tensors]
        max_len = max(lens)

        matrix = np.zeros((len(emb_tensors), max_len, emb_tensors[0].shape[1]))

        for i in range(len(emb_tensors)):
            matrix[i, :lens[i], :] = emb_tensors[i]

        return torch.FloatTensor(matrix), lens
