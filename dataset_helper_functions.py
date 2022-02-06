import os
import os.path as p
import pandas as pd

from sklearn.model_selection import train_test_split
from definitions import POLIT_DATA_DIR_PATH


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


def sample_development_set(devsize: float = 0.1, randstate: int = 2) -> None:
    """
    Create subset for development process.
    """
    train_path = p.join(POLIT_DATA_DIR_PATH, 'train')

    train_df = pd.read_csv(
        p.join(train_path, 'train_combined.tsv'),
        sep='\t'
    )

    _, tdmask = train_test_split(train_df['id'].values, test_size=devsize, random_state=randstate)

    train_dev_df = train_df[train_df['id'].isin(tdmask)].reset_index(drop=True)

    train_dev_df.to_csv(
        p.join(train_path, 'dev.tsv'),
        sep='\t',
        index=False
    )
