import torch
import nltk
import spacy
import pandas as pd
import numpy as np

from io import UnsupportedOperation
from definitions import *


class DebatesDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data: pd.DataFrame = None,
        file_path: str = None,
        transform=None,
    ) -> None:
        super(DebatesDataset, self).__init__()

        self.transform = transform

        if data is not None:
            self.data = data
        elif file_path is not None:
            suffix = file_path.split('.')[-1]
            if suffix == 'tsv':
                self.data = pd.read_csv(file_path, sep='\t', index_col=False)
            elif suffix == 'csv':
                self.data = pd.read_csv(file_path, index_col=False)
            elif suffix == 'pkl':
                self.data = pd.read_pickle(file_path)
            else:
                raise UnsupportedOperation("DebatesDataset only supports .tsv, .csv and .pkl formats.")
        else:
            raise Exception("Missing one of the arguments data, file_path.")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        sub = self.data.loc[index, :]

        if self.transform:
            return self.transform((sub.id, sub.content, sub.label, [], np.array([])))

        return (sub.id, sub.content, sub.label, np.array([]))
