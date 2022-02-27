import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from BertEmbedding import BertEmbedding
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTM(nn.Module):

    def __init__(
        self,
        device,
        hidden_dim=128,
        emb_pooling_strat='second_last',
        emb_from_pretrained='bert-base-uncased',
    ):
        super(BiLSTM, self).__init__()

        self.embedding = BertEmbedding(
            pooling_strat=emb_pooling_strat,
            from_pretrained=emb_from_pretrained,
            device=device
        )
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            self.embedding.dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.drop = nn.Dropout(p=0.5)
        self.dense = nn.Linear(2*hidden_dim, 1)

    def forward(self, sentences):
        embeddings = self.embedding(sentences)

        padded, lens = self._pad(embeddings)

        packed = pack_padded_sequence(padded, lens, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(packed_out, batch_first=True)

        out_forward = output[:, -1, :self.hidden_dim]
        out_reverse = output[:, -1, self.hidden_dim:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        out_drop = self.drop(out_reduced)

        out_dense = torch.squeeze(self.dense(out_drop), 1)
        pred = torch.sigmoid(out_dense)

        return pred

    @staticmethod
    def _pad(emb_tensors):
        lens = [len(s) for s in emb_tensors]
        max_len = max(lens)

        matrix = np.zeros((len(emb_tensors), max_len, emb_tensors[0].shape[1]))

        for i in range(len(emb_tensors)):
            matrix[i, :lens[i], :] = emb_tensors[i]

        return torch.FloatTensor(matrix), lens
