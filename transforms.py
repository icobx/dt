import os
import torch
import spacy
import nltk
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from definitions import *
from model_helper_functions import scale

# TODO: remove comments


class HandleStopwords(object):

    def __init__(self, stopwords: str = 'wstop', spacy_core: str = 'en_core_web_lg',) -> None:
        self.remove_stopwords = stopwords != 'wstop'
#         spacy.prefer_gpu()
        self.spacy = spacy.load(os.path.join(SPACY_MODEL_PATH, spacy_core))
        nltk.download('punkt')
        self.stopwords = set(nltk.corpus.stopwords.words('english'))

    def __call__(self, sample):
        sid, content, label, spacied, feature = sample

        if self.remove_stopwords:
            no_stop = ' '.join([w for w in nltk.word_tokenize(content) if w not in self.stopwords])
            spacied = [self.spacy(no_stop)]
        else:
            spacied = [self.spacy(content)]

        return sid, content, label, spacied, feature


class OneHot(object):

    def __init__(
        self,
        feat_type: str,
        from_selection: bool = False,
        stopwords: str = 'wstop',
    ) -> None:

        self.feat_type = feat_type
        self.remove_stopwords = stopwords != 'wstop'

        if from_selection:
            self.tags = FEATURE_SELECTION[feat_type][stopwords]
        else:
            self.tags = {
                'pos': SPACY_POS_TAGS,
                'tag': SPACY_FINE_POS_TAGS,
                'dep': SPACY_DEP_TAGS
            }[feat_type]

        self.enc = OneHotEncoder(handle_unknown='ignore')

        self.enc.fit(np.array(self.tags).reshape(-1, 1))

    def __call__(self, sample):
        ftwu = f'{self.feat_type}_'

        sid, content, label, spacied, feature = sample

        tags = [getattr(t, ftwu) for t in spacied[0]]
        
        one_hot = self.enc.transform(np.array(tags).reshape(-1, 1))
        one_hot = np.sum(one_hot, axis=0)
        one_hot = np.where(one_hot >= 1, 1, 0)

        feature = np.append(feature, one_hot[0])

        return sid, content, label, spacied, feature


class Sum(object):

    def __init__(
        self,
        feat_type: str,
        from_selection: bool = False,
        stopwords: str = 'wstop',
    ) -> None:

        self.feat_type = feat_type
        self.remove_stopwords = stopwords != 'wstop'

        if from_selection:
            # self.tags = [FEATURE_SELECTION[ft][stopwords] for ft in feat_type]
            self.tags = FEATURE_SELECTION[feat_type][stopwords]
        else:
            self.tags = {
                'pos': SPACY_POS_TAGS,
                'tag': SPACY_FINE_POS_TAGS,
                'dep': SPACY_DEP_TAGS
            }[feat_type]
            # self.tags = [{
            #     'pos': SPACY_POS_TAGS,
            #     'tag': SPACY_FINE_POS_TAGS,
            #     'dep': SPACY_DEP_TAGS
            # }[ft] for ft in feat_type]

        # self.lb = {ft: LabelBinarizer() for ft in feat_type}
        self.enc = OneHotEncoder(handle_unknown='ignore')

        # for i, lb in enumerate(self.lb):
        #     self.lb[lb].fit(self.tags[i])
        self.enc.fit(np.array(self.tags).reshape(-1, 1))

    def __call__(self, sample):
        # ftwu = [f'{ft}_' for ft in self.feat_type]
        ftwu = f'{self.feat_type}_'

        sid, content, label, spacied, feature = sample

        tags = [getattr(t, ftwu) for t in spacied[0]]

        feat_sum = self.enc.transform(np.array(tags).reshape(-1, 1))
        feat_sum = np.sum(feat_sum, axis=0)

        feature = np.append(feature, feat_sum[0])
        # for fta in ftwu:
        # tags =

        # x = self.lb[fta[:-1]].transform(tags)

        # x = np.sum(x, axis=0)
        # x = np.where(x >= 1, 1, 0)

        # feature.extend(x.tolist())

        return sid, content, label, spacied, feature


class CountWords(object):

    def __init__(self) -> None:
        pass

    def __call__(self, sample):
        sid, content, label, spacied, feature = sample

        word_count = len([t for t in spacied[0] if not t.is_punct])
        feature = np.append(feature, word_count)

        return sid, content, label, spacied, feature


class Scale(object):

    def __init__(self, min_r: float = 0.0, max_r: float = 1.0) -> None:
        self.min_r = min_r
        self.max_r = max_r

    def __call__(self, sample):
        sid, content, label, spacied, feature = sample

        feature = scale(feature, min_r=self.min_r, max_r=self.max_r)

        return sid, content, label, spacied, feature


class NoTransform(object):
    """
    Do nothing transform. Useful for feature optimization as interface unification.
    """

    def __init__(self, feat_type=None, from_selection=None, stopwords=None) -> None:
        pass

    def __call__(self, sample):
        return sample


class ToBinary(object):

    def __init__(self, n_bits: int = 6) -> None:
        self.n_bits = n_bits

    def __call__(self, sample):
        sid, content, label, spacied, feature = sample

        if len(feature) == 0:
            return sid, content, label, spacied, feature

        feature = feature if torch.is_tensor(feature) else torch.tensor(feature)
        feature = feature.int()

        mask = 2**torch.arange(self.n_bits-1, -1, -1).to(feature.device, feature.dtype)

        feature = feature.unsqueeze(-1).bitwise_and(mask).ne(0).byte().view(-1)

        return sid, content, label, spacied, feature


class ToTensor(object):

    def __init__(self) -> None:
        pass

    def __call__(self, sample):
        sid, content, label, _, feature = sample

        # if len(feature) == 0:
        #     return sid, content, label, torch.empty((1, ))

        feature = feature.float() if torch.is_tensor(feature) else torch.FloatTensor(feature)

        return sid, content, label, feature
