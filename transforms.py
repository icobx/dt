import torch
import spacy
import numpy as np

from sklearn.preprocessing import LabelBinarizer
from definitions import *
from model_helper_functions import scale

# TODO: remove comments


class OneHot(object):

    def __init__(
        self,
        feat_type: str,
        from_selection: bool = False,
        stopwords: str = 'wstop',
        spacy_core: str = 'en_core_web_lg'
    ) -> None:

        self.feat_type = feat_type
        self.stopwords = stopwords == 'wstop'

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
        self.lb = LabelBinarizer()

        # for i, lb in enumerate(self.lb):
        #     self.lb[lb].fit(self.tags[i])
        self.lb.fit(self.tags)

        self.spacy = spacy.load(spacy_core)
        # self.stopwords = set(nltk.corpus.stopwords.words('english'))

    def __call__(self, sample):
        # ftwu = [f'{ft}_' for ft in self.feat_type]
        ftwu = f'{self.feat_type}_'

        sid, content, label, spacied, feature = sample

        spacied = spacied if len(spacied) else [self.spacy(content)]

        tags = [getattr(t, ftwu) for t in spacied[0]]

        one_hot = self.lb.transform(tags)
        one_hot = np.sum(one_hot, axis=0)
        one_hot = np.where(one_hot >= 1, 1, 0)

        feature.extend(one_hot.tolist())
        # for fta in ftwu:
        # tags =

        # x = self.lb[fta[:-1]].transform(tags)

        # x = np.sum(x, axis=0)
        # x = np.where(x >= 1, 1, 0)

        # feature.extend(x.tolist())

        return sid, content, label, spacied, feature


class Sum(object):

    def __init__(
        self,
        feat_type: str,
        from_selection: bool = False,
        stopwords: str = 'wstop',
        spacy_core: str = 'en_core_web_lg'
    ) -> None:

        self.feat_type = feat_type
        self.stopwords = stopwords == 'wstop'

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
        self.lb = LabelBinarizer()

        # for i, lb in enumerate(self.lb):
        #     self.lb[lb].fit(self.tags[i])
        self.lb.fit(self.tags)

        self.spacy = spacy.load(spacy_core)
        # self.stopwords = set(nltk.corpus.stopwords.words('english'))

    def __call__(self, sample):
        # ftwu = [f'{ft}_' for ft in self.feat_type]
        ftwu = f'{self.feat_type}_'

        sid, content, label, spacied, feature = sample

        spacied = spacied if len(spacied) else [self.spacy(content)]

        tags = [getattr(t, ftwu) for t in spacied[0]]

        feat_sum = self.lb.transform(tags)
        feat_sum = np.sum(feat_sum, axis=0)

        feature.extend(feat_sum.tolist())
        # for fta in ftwu:
        # tags =

        # x = self.lb[fta[:-1]].transform(tags)

        # x = np.sum(x, axis=0)
        # x = np.where(x >= 1, 1, 0)

        # feature.extend(x.tolist())

        return sid, content, label, spacied, feature


class Scale(object):

    def __init__(self) -> None:
        pass

    def __call__(self, sample):
        sid, content, label, spacied, feature = sample

        feature = scale(np.asarray(feature), np.min(feature), np.max(feature))

        return sid, content, label, spacied, feature


class ToTensor(object):

    def __init__(self) -> None:
        pass

    def __call__(self, sample):
        sid, content, label, _, feature = sample

        feature = feature if torch.is_tensor(feature) else torch.tensor(feature)

        return sid, content, label, feature
