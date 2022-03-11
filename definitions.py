import os.path as p

# home
# DATA_DIR_PATH = '/Users/icobx/Documents/skola/dp/code/data'
# EXP_DIR_PATH = p.abspath('./exp')
# LOG_DIR_PATH = p.abspath('./log')

# gpu
DATA_DIR_PATH = '/home/jovyan/sharedstorage/s12b3v/dp/data'
EXP_DIR_PATH = '/home/jovyan/sharedstorage/s12b3v/dp/exp'
# LOG_DIR_PATH = p.abspath('./log')


PROC_DATA_DIR_PATH = p.join(DATA_DIR_PATH, 'dt-processed')

POLIT_DATA_DIR_PATH = p.join(DATA_DIR_PATH, 'political_debates')
COVID_DATA_DIR_PATH = p.join(DATA_DIR_PATH, 'covid_tweets')

SPACY_POS_TAGS = [
    "ADJ", "ADP", "ADV",
    "AUX", "CONJ", "CCONJ",
    "DET", "INTJ", "NOUN",
    "NUM", "PART", "PRON",
    "PROPN", "PUNCT", "SCONJ",
    "SYM", "VERB", "X", "SPACE"
]

SPACY_FINE_POS_TAGS = [
    '$', "''", ',', '-LRB-', '-RRB-', '.', ':', 'ADD', 'AFX',
    'CC', 'CD', 'DT', 'EX', 'FW', 'HYPH', 'IN', 'JJ', 'JJR',
    'JJS', 'LS', 'MD', 'NFP', 'NN', 'NNP', 'NNPS', 'NNS',
    'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP',
    'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',
    'VBZ', 'WDT', 'WP', 'WP$', 'WRB', 'XX', '``'
]

SPACY_DEP_TAGS = [
    'ROOT', 'acl', 'acomp', 'advcl', 'advmod', 'agent', 'amod',
    'appos', 'attr', 'aux', 'auxpass', 'case', 'cc', 'ccomp',
    'compound', 'conj', 'csubj', 'csubjpass', 'dative', 'dep',
    'det', 'dobj', 'expl', 'intj', 'mark', 'meta', 'neg', 'nmod',
    'npadvmod', 'nsubj', 'nsubjpass', 'nummod', 'oprd', 'parataxis',
    'pcomp', 'pobj', 'poss', 'preconj', 'predet', 'prep', 'prt',
    'punct', 'quantmod', 'relcl', 'xcomp'
]


FEATURE_SELECTION = {
    'pos': {
        'wstop': ['NUM', 'PROPN', 'NOUN', 'ADJ'],
        'wostop': ['NUM', 'SYM'],
    },
    'tag': {
        'wstop': ['CD', 'NNP', 'VBG', 'NN', 'NNS', 'JJ'],
        'wostop': ['CD', '$']
    },
    'dep': {
        'wstop': ['pobj', 'compound', 'npadvmod', 'amod', 'nummod'],
        'wostop': ['pcomp', 'quantmod', 'nummod', 'nsubjpass']
    }
}
