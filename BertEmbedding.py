from DebatesDataset import DebatesDataset
from torch.utils.data.dataloader import DataLoader
import os
import pandas as pd
import torch
import spacy
import numpy as np

from typing import List, Tuple
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelBinarizer
from definitions import SPACY_POS_TAGS, SPACY_DEP_TAGS, PROC_DATA_DIR_PATH, POLIT_DATA_DIR_PATH
from model_helper_functions import scale


class BertEmbedding():

    def __init__(
        self,
        pooling_strat: str,
        from_pretrained: str,
        device: torch.Tensor.device,
        spacy_core='en_core_web_lg',
        dep_features=False,
        triplet_features=False,
        scale=False
    ) -> None:
        self.pooling_strat = pooling_strat
        self.dev = device

        self.tokenizer = BertTokenizer.from_pretrained(from_pretrained)
        self.model = BertModel.from_pretrained(from_pretrained, output_hidden_states=True)
        self.model.eval()

        self.scale = scale

        dep_features = dep_features if dep_features else triplet_features
        self.dep_features = dep_features
        self.triplet_features = triplet_features
        self.spacy = None
        self.dep_binarizer = None
        self.pos_binarizer = None
        self.spacy_dim = 0
        if dep_features:
            self.spacy = spacy.load(spacy_core)
            self.spacy_dim = len(SPACY_DEP_TAGS)

            self.dep_binarizer = LabelBinarizer()
            self.dep_binarizer.fit(SPACY_DEP_TAGS)

            if triplet_features:
                self.spacy_dim += 2*len(SPACY_POS_TAGS)
                self.pos_binarizer = LabelBinarizer()
                self.pos_binarizer.fit(SPACY_POS_TAGS)

        self.pooling_strat_methods = {
            'last_four': BertEmbedding._ps_last_four,
            'last_four_sum': BertEmbedding._ps_last_four_sum,
            'second_last': BertEmbedding._ps_second_last,
        }
        self.pooling_strat_dims = {
            'last_four': 3072,
            'last_four_sum': 768,
            'second_last': 768,
        }
        self.dim = self.pooling_strat_dims[pooling_strat]
        """
        min last four:  tensor(-32.5030)
        min second last:  tensor(-32.5030)
        max last four:  tensor(10.1765)
        max second last:  tensor(8.0412)
        """

    def __call__(self, sentences: List[str]) -> Tuple[torch.Tensor, np.ndarray]:
        # tokenize the sentence
        tokenizer_output_dict = self.tokenizer(
            sentences,
            add_special_tokens=True,
            padding='longest',
            return_tensors='pt',
        )
        am_sum = torch.sum(tokenizer_output_dict['attention_mask'], dim=1)
        lengths = np.asarray([am_sum[i].item() for i in range(len(sentences))])

        tokenizer_output_dict['token_type_ids'] = tokenizer_output_dict['token_type_ids'] + 1

        with torch.no_grad():
            # Evaluating the model will return a different number of objects based on
            # how it's  configured in the `from_pretrained` call earlier. In this case,
            # becase we set `output_hidden_states = True`, the third item will be the
            # hidden states from all layers. See the documentation for more details:
            # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
            hidden_states = self.model(**tokenizer_output_dict)[2]
            pooled = self.pooling(hidden_states=hidden_states)

            if self.scale:
                pooled = scale(pooled)

            if not self.dep_features and not self.triplet_features:
                return pooled, lengths

            emb_w_feat = torch.cat(
                (
                    pooled,
                    self.create_features(sentences, tokenizer_output_dict['input_ids'])
                ),
                dim=2
            )
            return emb_w_feat, lengths+self.spacy_dim

    def create_features(self, sentences, input_ids):
        special_tokens = {'[CLS]', '[SEP]', '[PAD]'}
        features = []
        for i in range(len(sentences)):
            tokenized = self.tokenizer.convert_ids_to_tokens(input_ids[i])
            # spacy on joined tokenized to keep the same length
            tokenized_for_spacy = [t for t in tokenized if t not in special_tokens]
            spacied = self.spacy(' '.join(tokenized_for_spacy))
            encoded = []
            spacy_ind = 0

            for token in tokenized:
                if token in special_tokens:
                    encoded.append([0]*self.spacy_dim)

                elif spacy_ind < len(spacied):
                    word_ft = self.dep_binarizer.transform([spacied[spacy_ind].dep_]).tolist()[0]

                    if self.triplet_features:
                        token_pos = self.pos_binarizer.transform([spacied[spacy_ind].pos_]).tolist()[0]
                        head_pos = self.pos_binarizer.transform([spacied[spacy_ind].head.pos_]).tolist()[0]
                        word_ft = [*token_pos, *word_ft, *head_pos]

                    encoded.append(word_ft)

                    if token[:2] != '##':
                        spacy_ind += 1

            features.append(encoded)

        return torch.tensor(features)

    def pooling(self, hidden_states: torch.Tensor) -> torch.Tensor:
        stacked = torch.stack(hidden_states, dim=0)
        rearranged = stacked.permute(1, 2, 0, 3)

        return self.pooling_strat_methods[self.pooling_strat](rearranged)

    @staticmethod
    def _ps_last_four(hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                hidden_states[:, :, -1, :],
                hidden_states[:, :, -2, :],
                hidden_states[:, :, -3, :],
                hidden_states[:, :, -4, :],
            ),
            dim=2
        )

    @staticmethod
    def _ps_last_four_sum(hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.sum(hidden_states[:, :, -4:, :], dim=2)

    @staticmethod
    def _ps_second_last(hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states[:, :, -2, :]


# be = BertEmbedding('second_last', 'bert-base-uncased', torch.device('cpu'), triplet_features=False, scale=True)
# xx, _ = be(["That's false.", "This is a republican debate, right?"])
