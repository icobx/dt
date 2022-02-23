import os
import os.path as p
import requests
import json
import pandas as pd

from definitions import POLIT_DATA_DIR_PATH


label_threshold = 0.5

api_key = 'b972ba86d067455e909adc479fd9bf2b'
# cannot use /sentences/ endpoint, because their sentence tokenization yields different number of sentences
api_endpoint = "https://idir.uta.edu/claimbuster/api/v2/score/text/sentences/"
request_headers = {"x-api-key": api_key}

path = p.join(POLIT_DATA_DIR_PATH, 'train_weak')
files = [f for f in os.listdir(path) if '_combined' not in f]

for file in files:
    df = pd.read_csv(
        p.join(path, file),
        sep='\t',
        index_col=False,
        names=['id', 'src', 'content', 'label']
    )
    sentences = df['content'].values
    input_text = ' '.join(sentences)
    api_response = requests.post(url=api_endpoint, headers=request_headers, json={"input_text": input_text})
    results = api_response.json()['results']

    sent_count = 0
    for i in range(len(sentences)):
        res_len = len(results[i]['text'])
        sent_len = len(sentences[i])
        if res_len != sent_len:
            print(sentences[i])
            print(results[i]['text'])
            print('\n')
            sent_count += 1

        if sent_count == 10:
            exit()
    # scores = []
    # for i, sentence in enumerate(df['content'].values):
    #     print(i)

    #     score = api_response.json()['results'][0]['score']

    #     scores.append(1 if score > label_threshold else 0)

    # print(scores)
    # print(len(scores))
    # print(df.shape)
    exit()

# input_claim = "Now, I also say that as far as Senator Kennedy's proposals are concerned, that, again, the question is not simply one of uh - a presidential veto stopping programs."
# # Define the endpoint (url) with the claim formatted as part of it, api-key (api-key is sent as an extra header)
# api_endpoint = f"https://idir.uta.edu/claimbuster/api/v2/score/text/"


# # Print out the JSON payload the API sent back
# print(api_response.json())

# df = pd.read_csv(
#     p.join(POLIT_DATA_DIR_PATH, 'train_weak', '19600926_presidential_chicago.tsv'),
#     sep='\t',
#     index_col=False,
#     names=['id', 'src', 'content', 'label']
# )
# input_text = ' '.join(df['content'].values)
# # print(input_text)
# # exit()
# print(df.shape)
# Send the POST request to the API and store the api response
# api_response = requests.post(url=api_endpoint, json=payload, headers=request_headers)

# # Print out the JSON payload the API sent back
# print(len([1 if x['score'] > 0.5 else 0 for x in api_response.json()['results']]))
# print()
