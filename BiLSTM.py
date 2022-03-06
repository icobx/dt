import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from BertEmbedding import BertEmbedding
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.preprocessing import LabelBinarizer
# from definitions import
# from model_helper_functions import create_features


class BiLSTM(nn.Module):

    def __init__(
        self,
        device,
        dropout=0.5,
        hidden_dim=128,
        emb_pooling_strat='second_last',
        emb_from_pretrained='bert-base-uncased',
        word_level_dep_features=False,
        word_level_triplet_features=False,
    ):
        super(BiLSTM, self).__init__()

        self.embedding = BertEmbedding(
            pooling_strat=emb_pooling_strat,
            from_pretrained=emb_from_pretrained,
            device=device,
            spacy_core='en_code_web_lg',
            dep_features=word_level_dep_features,
            triplet_features=word_level_triplet_features
        )

        self.hidden_dim = hidden_dim
        self.sent_level_feature_dim = self.embedding.spacy_dim
        self.lstm = nn.LSTM(
            self.embedding.dim+self.embedding.spacy_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.drop = nn.Dropout(p=dropout)
        self.dense = nn.Linear((2*hidden_dim)+self.embedding.spacy_dim, 1)

    def forward(self, sentences):
        embeddings, lengths = self.embedding(sentences)

        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(packed_out, batch_first=True)

        out_forward = output[:, -1, :self.hidden_dim]
        out_reverse = output[:, -1, self.hidden_dim:]
        out = torch.cat((out_forward, out_reverse), 1)

        out_drop = self.drop(out)

        out_dense = torch.squeeze(self.dense(out_drop), 1)
        pred = torch.sigmoid(out_dense)

        return pred

    @staticmethod
    def _append_features(x, features, cat_dim):
        if features is None:
            return x

        features = torch.FloatTensor(features)

        return torch.cat((x, features), dim=cat_dim)

    @staticmethod
    def _pad(emb_tensors):
        lens = [len(s) for s in emb_tensors]
        max_len = max(lens)

        matrix = np.zeros((len(emb_tensors), max_len, emb_tensors[0].shape[1]))

        for i in range(len(emb_tensors)):
            matrix[i, :lens[i], :] = emb_tensors[i]

        return torch.FloatTensor(matrix), lens


# bl = BiLSTM(torch.device('cpu'))

# x = ['This is a sentence', 'Completely different topic we are talking about here.']
# q = [[0, 1, 1, 0, 0, 0, 1, 0, 1], [0, 1, 1, 0, 0, 0, 1, 0, 1]]
# res = bl(x)
