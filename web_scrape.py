import os.path as p
import re
import requests
import nltk
import pandas as pd
from tqdm import tqdm

from bs4 import BeautifulSoup
from definitions import POLIT_DATA_DIR_PATH


# GET DEBATES
base_url = 'https://www.presidency.ucsb.edu'
debates_url = 'people/other/presidential-candidate-debates'

label_threshold = 0.5

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

for i in tqdm(range(4, 5), desc='pages'):
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


# debate_r = requests.get(
#     'https://www.presidency.ucsb.edu/documents/democratic-candidates-all-american-presidential-forum-howard-university-washington-dc')
# debate_soup = BeautifulSoup(debate_r.content, 'html.parser')

# paragraphs = debate_soup.find('div', class_='field-docs-content').find_all('p')

# debate_df = pd.DataFrame()
# for par in paragraphs:
#     text = par.text
#     sentences = nltk.sent_tokenize(par.text)
#     mtch = pattern.match(par.text)

#     if mtch is not None:
#         end = mtch.end()
#         tbsrc = text[:end-1]
#         if tbsrc.lower() in to_skip:
#             continue

#         src = tbsrc
#         print(src)
#         text = text[end:]

#     src = [src]*len(sentences)
#     print(src)
