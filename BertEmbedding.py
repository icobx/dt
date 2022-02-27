from lib2to3.pgen2 import token
import torch

from typing import List
from transformers import BertTokenizer, BertModel


class BertEmbedding():

    def __init__(self, pooling_strat: str, from_pretrained: str, device: torch.Tensor.device) -> None:
        self.pooling_strat = pooling_strat
        self.dev = device

        self.tokenizer = BertTokenizer.from_pretrained(from_pretrained)
        self.model = BertModel.from_pretrained(from_pretrained, output_hidden_states=True)
        self.model.eval()

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

    def __call__(self, sentences: List[str]) -> List[torch.Tensor]:
        # add necessary tokens
        sent_added = [f'[CLS] {s} [SEP]' for s in sentences]
        # tokenize the sentence
        # sent_tokenized = [self.tokenizer.tokenize(s) for s in sent_added]
        tokenizer_output_dict = self.tokenizer(sent_added, padding='longest', return_tensors='pt')
        tokenizer_output_dict['token_type_ids'] = tokenizer_output_dict['token_type_ids'] + 1
        print(tokenizer_output_dict)
        # longest = tokenizer_output_dict['input_ids']
        # map tokens to vocab. indices
        # tokens_indexed = [self.tokenizer.convert_tokens_to_ids(st) for st in sent_tokenized]
        # create segments layer
        # segments_layer = [[1] * len(s) for s in sent_tokenized]
        # flatten and convert to tensors
        # TODO: .to(self.dev) might cause problems
        # tokens_tensor = torch.tensor([[windex for idx in tokens_indexed for windex in idx]]).to(self.dev)
        # segments_tensor = torch.tensor([[sindex for segment in segments_layer for sindex in segment]]).to(self.dev)
        # tokens_tensor = tokenizer_output_dict['input_ids']
        with torch.no_grad():
            # Evaluating the model will return a different number of objects based on
            # how it's  configured in the `from_pretrained` call earlier. In this case,
            # becase we set `output_hidden_states = True`, the third item will be the
            # hidden states from all layers. See the documentation for more details:
            # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
            hidden_states = self.model(**tokenizer_output_dict)[2]
            # Concatenate the tensors for all layers. We use `stack` here to
            # create a new dimension in the tensor.
            token_embeddings = torch.stack(hidden_states, dim=0)
            # Remove dimension 1, the "batches".
            print(token_embeddings.shape)
            # token_embeddings = torch.squeeze(token_embeddings, dim=1)
            # Swap dimensions 0 and 1. 0 1 2 3
            token_embeddings = token_embeddings.permute(1, 2, 0, 3)
            print(token_embeddings.shape)

        # pooled = torch.stack([self.pooling(token) for token in token_embeddings])
        # TODO: FIX POOLING, 4 DIMS instead of 3
        # embeddings = []
        # prev = 0
        # for sent in sent_tokenized:
        #     embeddings.append(pooled[prev:prev+len(sent), :])
        #     prev += len(sent)

        return token_embeddings

    def pooling(self, token: torch.Tensor) -> torch.Tensor:
        return self.pooling_strat_methods[self.pooling_strat](token)

    @staticmethod
    def _ps_last_four(token: torch.Tensor) -> torch.Tensor:
        return torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)

    @staticmethod
    def _ps_last_four_sum(token: torch.Tensor) -> torch.Tensor:
        return torch.sum(token[-4:], dim=0)

    @staticmethod
    def _ps_second_last(token: torch.Tensor) -> torch.Tensor:
        return token[-2]


be = BertEmbedding('second_last', 'bert-base-uncased', torch.device('cpu'))

# be()
# tokenizer = be.tokenizer
x = ['Hello, this is sentence.', 'And this is much longer sentence, because it is comprised of two']
# # print()
# ret = tokenizer(x, return_tensors='pt', padding='longest')
# retep = tokenizer.encode_plus(
#     x,
#     add_special_tokens=True,
#     truncation=True,
#     padding="max_length",
#     return_attention_mask=True,
#     return_tensors="pt"
# )

# print(ret)
# print()
# print(retep)
y = be(x)

print(y)


# import torch

# from typing import List
# from transformers import BertTokenizer, BertModel


# class BertEmbedding():

#     def __init__(self, pooling_strat: str, from_pretrained: str, device: torch.Tensor.device) -> None:
#         self.pooling_strat = pooling_strat
#         self.dev = device

#         self.tokenizer = BertTokenizer.from_pretrained(from_pretrained)
#         self.model = BertModel.from_pretrained(from_pretrained, output_hidden_states=True)
#         self.model.eval()

#         self.pooling_strat_methods = {
#             'last_four': BertEmbedding._ps_last_four,
#             'last_four_sum': BertEmbedding._ps_last_four_sum,
#             'second_last': BertEmbedding._ps_second_last,
#         }
#         self.pooling_strat_dims = {
#             'last_four': 3072,
#             'last_four_sum': 768,
#             'second_last': 768,
#         }
#         self.dim = self.pooling_strat_dims[pooling_strat]

#     def __call__(self, sentences: List[str]) -> List[torch.Tensor]:
#         # add necessary tokens
#         sent_added = [f'[CLS] {s} [SEP]' for s in sentences]
#         # tokenize the sentence
#         sent_tokenized = [self.tokenizer.tokenize(s) for s in sent_added]
#         print(type(sent_tokenized[0]))
#         # map tokens to vocab. indices
#         tokens_indexed = [self.tokenizer.convert_tokens_to_ids(st) for st in sent_tokenized]
#         # create segments layer
#         segments_layer = [[1] * len(s) for s in sent_tokenized]
#         # flatten and convert to tensors
#         # TODO: .to(self.dev) might cause problems
#         tokens_tensor = torch.tensor([[windex for idx in tokens_indexed for windex in idx]]).to(self.dev)
#         segments_tensor = torch.tensor([[sindex for segment in segments_layer for sindex in segment]]).to(self.dev)

#         with torch.no_grad():
#             # Evaluating the model will return a different number of objects based on
#             # how it's  configured in the `from_pretrained` call earlier. In this case,
#             # becase we set `output_hidden_states = True`, the third item will be the
#             # hidden states from all layers. See the documentation for more details:
#             # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
#             hidden_states = self.model(tokens_tensor, segments_tensor)[2]
#             # Concatenate the tensors for all layers. We use `stack` here to
#             # create a new dimension in the tensor.
#             token_embeddings = torch.stack(hidden_states, dim=0)
#             # Remove dimension 1, the "batches".
#             token_embeddings = torch.squeeze(token_embeddings, dim=1)
#             # Swap dimensions 0 and 1.
#             token_embeddings = token_embeddings.permute(1, 0, 2)

#         pooled = torch.stack([self.pooling(token) for token in token_embeddings])

#         embeddings = []
#         prev = 0
#         for sent in sent_tokenized:
#             embeddings.append(pooled[prev:prev+len(sent), :])
#             prev += len(sent)

#         return embeddings

#     def pooling(self, token: torch.Tensor) -> torch.Tensor:
#         return self.pooling_strat_methods[self.pooling_strat](token)

#     @staticmethod
#     def _ps_last_four(token: torch.Tensor) -> torch.Tensor:
#         return torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)

#     @staticmethod
#     def _ps_last_four_sum(token: torch.Tensor) -> torch.Tensor:
#         return torch.sum(token[-4:], dim=0)

#     @staticmethod
#     def _ps_second_last(token: torch.Tensor) -> torch.Tensor:
#         return token[-2]
