import os
import re
import os.path as p
import pandas as pd
import numpy as np
import requests

from tqdm import tqdm
from bs4 import BeautifulSoup
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

def combine_debates(splits=['test', 'train'], prepend_title='') -> None:
    """
    Combines debates from multiple .tsv files into one. Ids for each record are created by concatenating date of the
    debate and index of the record.
    """
    if len(prepend_title):
        prepend_title = f'{prepend_title}_'

    data_paths = [p.join(POLIT_DATA_DIR_PATH, s) for s in splits]
    # [
    #     p.join(POLIT_DATA_DIR_PATH, 'test'),
    #     p.join(POLIT_DATA_DIR_PATH, 'train')
    # ]
    is_weak = 'train_weak' in splits

    for path in data_paths:
        header = ['i', 'src', 'content', 'label']
        if is_weak:
            header.append('score')

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

            header_w_id = ['i', 'id', 'src', 'content', 'label']
            if is_weak:
                header_w_id.append('score')
            df = df[header_w_id]

            df_combined = df_combined.append(
                df,
                ignore_index=True,
            )

        df_combined.to_csv(
            p.join(path, f'{prepend_title}{path.split(os.sep)[-1]}_combined.tsv'),
            sep='\t',
            index=False,
        )


def create_validation_subset(valsize: float = 0.25, randstate: int = 22, should_return=False) -> None:
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

    if should_return:
        return train_df, val_df

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
    randstate: int = 2,
    should_return: bool = False,
    train_filename: str = None
) -> None:
    """
    Create subset for development process.
    """
    dev_path = p.join(PROC_DATA_DIR_PATH, 'dev')
    train_path = p.join(POLIT_DATA_DIR_PATH, 'train')
    train_filename = train_filename if train_filename else 'train_combined.tsv'

    train_df = pd.read_csv(
        p.join(train_path, train_filename),
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

    if should_return:
        return train_dev_df

    train_dev_df.to_csv(
        p.join(dev_path, 'dev.tsv'),
        sep='\t',
        index=False
    )


def scrape_n_label(label_threshold: float = 0.5) -> None:
    """
    Scrape The American Presidency Project (https://www.presidency.ucsb.edu) collection of presidential debates.
    """
    # GET DEBATES
    base_url = 'https://www.presidency.ucsb.edu'
    debates_url = 'people/other/presidential-candidate-debates'

    api_key = 'b972ba86d067455e909adc479fd9bf2b'
    api_endpoint = "https://idir.uta.edu/claimbuster/api/v2/score/text/sentences/"
    request_headers = {"x-api-key": api_key}

    pattern = re.compile('^[A-Z][a-zA-Z\.\,\ ]+:')
    to_skip = {
        'participant', 'participants',
        'moderator', 'moderators',
        'sponsor', 'sponsors',
        'panelist', 'panelists',
        'other', 'others'
    }
    ignore_in_name = {'in', 'debate', 'Debate'}

    for i in tqdm(range(5), desc='pages'):
        r = requests.get(f'{base_url}/{debates_url}?page={i}')
        soup = BeautifulSoup(r.content, 'html.parser')

        rows = soup.find_all('div', class_='views-row')

        for row in tqdm(rows, desc='debates', leave=False):
            ts = ''.join(row.span.get('content').split('T')[0].split('-'))
            name = '_'.join([
                w.strip(' .,:').lower() for w in row.a.text.split()
                if '\"' not in w and w not in ignore_in_name
            ])

            debate_url = row.a.get('href')
            debate_r = requests.get(f'{base_url}{debate_url}')
            debate_soup = BeautifulSoup(debate_r.content, 'html.parser')

            paragraphs = debate_soup.find('div', class_='field-docs-content').find_all('p')

            debate_df = pd.DataFrame()
            for par in tqdm(paragraphs, desc='paragraphs', leave=False):
                text = par.text
                # sentences = nltk.sent_tokenize(par.text)
                mtch = pattern.match(par.text)

                if mtch is not None:
                    end = mtch.end()
                    tbsrc = text[:end-1]
                    if tbsrc.lower() in to_skip:
                        continue

                    src = tbsrc
                    text = text[end:]

                api_response = requests.post(url=api_endpoint, headers=request_headers, json={"input_text": text})
                results = api_response.json()['results']

                src_values = [src]*len(results)
                sentences = [r['text'] for r in results]
                # binary labelling based on label_threshold
                label = [
                    1 if r['score'] > label_threshold else 0 for r in results
                ]
                score = [r['score'] for r in results]

                debate_df = debate_df.append(pd.DataFrame(list(zip(src_values, sentences, label, score))))

            debate_df[4] = [x for x in range(1, debate_df.shape[0]+1)]
            debate_df[[4, 0, 1, 2, 3]].to_csv(
                p.join(POLIT_DATA_DIR_PATH, 'train_weak', f'{ts}_{name}.tsv'),
                sep='\t',
                header=False,
                index=False
            )


def filter_by_length(df, length=3):
    """
    Drop sentences of length `length` or shorter.
    """
    import nltk

    def get_length(x):
        return len(nltk.word_tokenize(x))

    df['length'] = df['content'].apply(get_length)

    return df[df['length'] > length].reset_index(drop=True)


def pad_features(dataset):
    max_len = 0
    for _, _, _, feature in dataset:
        max_len = max(len(feature), max_len)

    for i in range(len(dataset)):
        a, b, c, d = dataset[i]
        feat_len = len(d)
        if feat_len < max_len:
            diff = max_len - feat_len
            print('diff', diff)

            padding = torch.zeros((diff, ))
            print('padding', padding)
            d = torch.cat((padding, d), dim=0)
            print('feature;', d)

        dataset[i] = (a, b, c, d)
        print(f'dataset at {i}', dataset[i])
    return dataset


# combine_debates()
# create_validation_subset()
# sample_development_set()
