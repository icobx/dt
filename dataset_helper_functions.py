import os
import os.path as p
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from definitions import POLIT_DATA_DIR_PATH, PROC_DATA_DIR_PATH


# relative locations
# covid-19 tweeets
# train: train/v1/training
# validation: train/v1/dev
# test: test/test-input/test-gold

# political debateds
# train: training/train_combined
# validation: val | random seed 22
# test: test/test_combined

def combine_debates() -> None:
    """
    Combines debates from multiple .tsv files into one. Ids for each record are created by concatenating date of the
    debate and index of the record.
    """

    data_paths = [
        p.join(POLIT_DATA_DIR_PATH, 'test'),
        p.join(POLIT_DATA_DIR_PATH, 'train')
    ]

    for path in data_paths:
        header = ['i', 'src', 'content', 'label']

        df_combined = pd.DataFrame()
        x = 0
        for f in os.listdir(path):
            filepath = p.join(path, f)

            if not p.isfile(filepath):
                continue

            fsplit = f.split('_')

            # skip previously combined
            if fsplit[-1] == 'combined.tsv':
                continue

            df = pd.read_csv(
                filepath,
                sep='\t',
                index_col=False,
                names=header
            )

            df = df[df['src'] != 'SYSTEM']

            debate_date = fsplit[0]

            df['id'] = df.apply(
                lambda x: f"{debate_date}{x['i']}",
                result_type='expand',
                axis='columns'
            )

            header_rear = ['i', 'id', 'src', 'content', 'label']
            df = df[header_rear]

            df_combined = df_combined.append(
                df,
                ignore_index=True,
            )

        df_combined.to_csv(
            p.join(path, f'{path.split(os.sep)[-1]}_combined.tsv'),
            sep='\t',
            index=False,
        )


def create_validation_subset(valsize: float = 0.25, randstate: int = 22) -> None:
    """
    Creates validation data by splitting training data. Ratio training/validation: 1-valsize/valsize.
    Can only be ran after combining of debates.
    """
    train_df = pd.read_csv(
        p.join(POLIT_DATA_DIR_PATH, 'train', 'train_combined.tsv'),
        sep='\t'
    )
    # TODO: rework this, only one class present in val
    # print(train_df.loc[train_df['label'] == 1].shape)

    tmask, vmask = train_test_split(train_df['id'].values, test_size=valsize, random_state=randstate)

    val_df = train_df[train_df['id'].isin(vmask)].reset_index(drop=True)

    train_df = train_df[train_df['id'].isin(tmask)].reset_index(drop=True)

    val_df.to_csv(
        p.join(POLIT_DATA_DIR_PATH, 'val', 'val_combined.tsv'),
        sep='\t',
        index=False,
    )
    train_df.to_csv(
        p.join(POLIT_DATA_DIR_PATH, 'train', 'train_combined.tsv'),
        sep='\t',
        index=False,
    )


def sample_development_set(
    dev_frac: float = 0.1,
    y_col_name: str = 'label',
    index_col_name: str = 'id',
    randstate: int = 2
) -> None:
    """
    Create subset for development process.
    """
    dev_path = p.join(PROC_DATA_DIR_PATH, 'dev')
    train_path = p.join(POLIT_DATA_DIR_PATH, 'train')

    train_df = pd.read_csv(
        p.join(train_path, 'train_combined.tsv'),
        sep='\t'
    )
    # get all classes
    y_groups = train_df[y_col_name].unique()

    # calculate size of one class
    size = int(train_df.shape[0]*dev_frac) // len(y_groups)

    # if calculated size is larger than occurances of given class change size to number of occurances of the class
    for label in y_groups:
        label_count = train_df.loc[train_df[y_col_name] == label, 'id'].shape[0]

        size = min(size, label_count)

    train_dev_df = pd.DataFrame()
    for label in y_groups:
        indexes = train_df.loc[train_df[y_col_name] == label, 'id'].values
        rand_indexes = np.random.choice(indexes, size=size, replace=False, )

        train_dev_df = train_dev_df.append(train_df.loc[train_df[index_col_name].isin(rand_indexes)])

    train_dev_df = train_dev_df.reset_index(drop=True)

    train_dev_df.to_csv(
        p.join(dev_path, 'dev.tsv'),
        sep='\t',
        index=False
    )


# create_validation_subset()
# sample_development_set()
