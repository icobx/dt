import torch

from transformers import BertTokenizer, BertModel


class BertEmbedding():

    def __init__(self, pooling_strat: str, from_pretrained: str) -> None:
        self.pooling_strat = pooling_strat

        self.tokenizer = BertTokenizer.from_pretrained(from_pretrained)
        self.model = BertModel.from_pretrained(from_pretrained, output_hidden_states=True)
        self.model.eval()

        self.pooling_strat_methods = {
            'last_four': BertEmbedding.__ps_last_four,
            'second_last': BertEmbedding.__ps_second_last
        }

    def transform(self, sentence: str) -> str:
        # add necessary tokens
        sent_added = f'[CLS] {sentence} [SEP]'
        # tokenize the sentence
        sent_tokenized = self.tokenizer.tokenize(sent_added)
        # map tokens to vocab. indices
        tokens_indexed = self.tokenizer.convert_tokens_to_ids(sent_tokenized)
        # create segments layer
        segments_layer = [1] * len(sent_tokenized)
        # convert to tensors
        tokens_tensor = torch.tensor([tokens_indexed])
        segments_tensor = torch.tensor([segments_layer])

        with torch.no_grad():
            # Evaluating the model will return a different number of objects based on
            # how it's  configured in the `from_pretrained` call earlier. In this case,
            # becase we set `output_hidden_states = True`, the third item will be the
            # hidden states from all layers. See the documentation for more details:
            # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
            hidden_states = self.model(tokens_tensor, segments_tensor)[2]
            # Concatenate the tensors for all layers. We use `stack` here to
            # create a new dimension in the tensor.
            token_embeddings = torch.stack(hidden_states, dim=0)
            # Remove dimension 1, the "batches".
            token_embeddings = torch.squeeze(token_embeddings, dim=1)
            # Swap dimensions 0 and 1.
            token_embeddings = token_embeddings.permute(1, 0, 2)

        embeddings = []
        for token in token_embeddings:
            embeddings.append(self.pooling(token))

        return embeddings

    def pooling(self, token: torch.Tensor) -> torch.Tensor:
        return self.pooling_strat_methods[self.pooling_strat](token)

    @staticmethod
    def __ps_last_four(token: torch.Tensor) -> torch.Tensor:
        return torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)

    @staticmethod
    def __ps_second_last(token: torch.Tensor) -> torch.Tensor:
        return token[-2]
