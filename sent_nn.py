import torch
import torch.nn as nn


class SentNN(nn.Module):

    def __init__(
        self,
        embeddings_dim=768,
        sentence_level_feature_dim=0,
        dropout=0.5,
#         hidden_dim=128,
#         n_hidden_layers=1
    ):
        super(SentNN, self).__init__()

#         dropouts = [nn.Dropout(p=dropout) for _ in range(n_hidden_layers+1)]
#         dense_layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)]

#         sequence = [None]*(len(dropouts)+len(dense_layers))

#         sequence[::2] = dropouts
#         sequence[1::2] = dense_layers

#         self.layers = nn.Sequential(nn.Linear(embeddings_dim+sentence_level_feature_dim, hidden_dim), *sequence, nn.Linear(hidden_dim, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.dense = nn.Linear(embeddings_dim+sentence_level_feature_dim, 1)
        

    def forward(self, embeddings, sent_level_features=None):
        out = self._append_features(embeddings, sent_level_features, cat_dim=1)
        
        out = self.dropout(out)
        out = self.dense(out)
        
        return torch.squeeze(out, 1)

    @staticmethod
    def _append_features(x, features, cat_dim):
        if features is None or features.size()[-1] == 0:
            return x
        
        return torch.cat((x, features), dim=cat_dim).float()


# st = SentenceTransformer('all-mpnet-base-v2')

# xx = st.encode(['You hired some workers from Poland...', 'This is a testing sentence.'], convert_to_tensor=True)

# print(xx.size())

# snn = SentNN(dropout=0.1)

# yy = snn(xx)
# print(yy.size())
# print(yy)
# print(torch.sigmoid(yy))
# res = {}
# for mn in ['all-mpnet-base-v2', 'all-MiniLM-L6-v2', 'multi-qa-mpnet-base-dot-v1']:
#     st = SentenceTransformer(mn)

#     xx = st.encode(['You hired some workers from Poland...'], convert_to_tensor=True)

#     res[mn] = xx.size()

# print(res)
