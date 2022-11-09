from multiprocessing import Pool

import librosa as lr
from variables import *
import utils
import pandas as pd
import numpy as np
from ast import literal_eval
import datetime
from tqdm import tqdm

import sklearn as skl
import sklearn.naive_bayes
import sklearn.neighbors
import sklearn.ensemble
import sklearn.tree
import sklearn.svm
# import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
# from sklearn.utils import shuffle
# from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, LabelBinarizer, StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC, LinearSVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# from sklearn.multiclass import OneVsRestClassifier
import pickle

def tratar_campo(campo):
    if not pd.isna(campo):
        campo = literal_eval(campo)
    return campo


def train_models(fe_functions, funcoes_de_ml, genres, tracks_ids, tracks):
    features = utils.load(FEATURE_DIR + '/features.csv')
    features_start = utils.load(FEATURE_DIR + '/features_start.csv')
    features_middle = utils.load(FEATURE_DIR + '/features_middle.csv')
    features_end = utils.load(FEATURE_DIR + '/features_end.csv')

    features_files = [
        features,
        features_start,
        features_middle,
        features_end,
    ]

    tuples = []
    for i in funcoes_de_ml:
        for j in range(4):
            tuples.append((i, j))

    colunas = pd.MultiIndex.from_tuples(tuples)

    scores = pd.DataFrame(index=genres['title'], columns=colunas).astype(object)

    for i, feature_file in enumerate(features_files):
        # if i == 0:
        #     continue
        for genre_id, genre in list(zip(genres.index, genres['title'])):
            train = tracks.index[tracks['set', 'split'] == 'training']
            test = tracks.index[tracks['set', 'split'] == 'test']

            results = tracks[('track', 'genres')].apply(lambda lst: 1 if genre_id in lst else 0)

            feature_train = feature_file[feature_file.index.isin(train)]
            feature_test = feature_file[feature_file.index.isin(test)]

            results_train = results[results.index.isin(train)]
            results_test = results[results.index.isin(test)]

            data = {
                'X_train': feature_train,
                'y_train': results_train,
                'X_test': feature_test,
                'y_test': results_test
            }

            for ml in funcoes_de_ml:
                try:
                    ml_algorithm = MachineLearning.training_method(ml)
                    model = ml_algorithm.train(data=data)
                    score = ml_algorithm.score

                    scores.loc[genre][(ml, i)] = score
                    print(f'{MODEL_DIR}/{genre}/{ml}/{i}.pkl')
                    with open(f'{MODEL_DIR}/{genre}/{ml}/{i}.pkl', 'wb') as file:
                        pickle.dumps(model, file)
                except:
                    pass

        scores.to_csv(FEATURE_DIR + '/scores.csv', encoding='utf-8')
















    # for i, feature in enumerate(features_files):
    #     for genre, id in music_ids.items():
    #         for ml in funcoes_de_ml:
    #             model = ml()
    #             features = data.loc[:, [funcao.__name__ for funcao in fe_functions]][list(music_ids.values())].tolist()
    #             results = [1 if i == genre else 0 for i in music_ids]
    #             model.fit(features, results)
    #

'''
## Treinando os modelos

full = tracks['set', 'subset'] <= 'large'

train = tracks['set', 'split'] == 'training'
val = tracks['set', 'split'] == 'validation'
test = tracks['set', 'split'] == 'test'

y_train = tracks.loc[full & train, ('track', 'genre_top')]
y_test = tracks.loc[full & test, ('track', 'genre_top')]
X_train = features.loc[full & train, 'mfcc']
X_test = features.loc[full & test, 'mfcc']

print('{} training examples, {} testing examples'.format(y_train.size, y_test.size))
print('{} features, {} classes'.format(X_train.shape[1], y_train.unique().size))

# Be sure training samples are shuffled.
X_train, y_train = skl.utils.shuffle(X_train, y_train, random_state=42)

# Standardize features by removing the mean and scaling to unit variance.
scaler = skl.preprocessing.StandardScaler(copy=False)
scaler.fit_transform(X_train)
scaler.transform(X_test)

# Support vector classification.
clf = skl.svm.SVC(random_state=42)
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print('Accuracy: {:.2%}'.format(score))
'''
#
# def pre_process(tracks, features, columns, multi_label=False, verbose=False):
#     if not multi_label:
#         # Assign an integer value to each genre.
#         enc = LabelEncoder()
#         labels = tracks['track', 'genre_top']
#         # y = enc.fit_transform(tracks['track', 'genre_top'])
#     else:
#         # Create an indicator matrix.
#         enc = MultiLabelBinarizer()
#         # labels = tracks['track', 'genres_all']
#         labels = tracks['track', 'genres']
#
#     train = tracks.index[tracks['set', 'split'] == 'training']
#     val = tracks.index[tracks['set', 'split'] == 'validation']
#     test = tracks.index[tracks['set', 'split'] == 'test']
#
#     # Split in training, validation and testing sets.
#     y_train = enc.fit_transform(labels[train])
#     y_val = enc.transform(labels[val])
#     y_test = enc.transform(labels[test])
#     X_train = features.loc[train, columns].as_matrix()
#     X_val = features.loc[val, columns].as_matrix()
#     X_test = features.loc[test, columns].as_matrix()
#
#     X_train, y_train = shuffle(X_train, y_train, random_state=42)
#
#     # Standardize features by removing the mean and scaling to unit variance.
#     scaler = StandardScaler(copy=False)
#     scaler.fit_transform(X_train)
#     scaler.transform(X_val)
#     scaler.transform(X_test)
#
#     return y_train, y_val, y_test, X_train, X_val, X_test
#
# def test_classifiers_features(classifiers, feature_sets, multi_label=False):
#     columns = list(classifiers.keys()).insert(0, 'dim')
#     scores = pd.DataFrame(columns=columns, index=feature_sets.keys())
#     times = pd.DataFrame(columns=classifiers.keys(), index=feature_sets.keys())
#     for fset_name, fset in tqdm_notebook(feature_sets.items(), desc='features'):
#         y_train, y_val, y_test, X_train, X_val, X_test = pre_process(tracks, features_all, fset, multi_label)
#         scores.loc[fset_name, 'dim'] = X_train.shape[1]
#         for clf_name, clf in classifiers.items():  # tqdm_notebook(classifiers.items(), desc='classifiers', leave=False):
#             t = time.process_time()
#             clf.fit(X_train, y_train)
#             score = clf.score(X_test, y_test)
#             scores.loc[fset_name, clf_name] = score
#             times.loc[fset_name, clf_name] = time.process_time() - t
#     return scores, times
#
# def format_scores(scores):
#     def highlight(s):
#         is_max = s == max(s[1:])
#         return ['background-color: yellow' if v else '' for v in is_max]
#     scores = scores.style.apply(highlight, axis=1)
#     return scores.format('{:.2%}', subset=pd.IndexSlice[:, scores.columns[1]:])



'''
classifiers = {
    #LogisticRegression(),
    'LR': OneVsRestClassifier(LogisticRegression()),
    'SVC': OneVsRestClassifier(SVC()),
    'MLP': MLPClassifier(max_iter=700),
}

feature_sets = {
    'mfcc': 'mfcc',
    'mfcc/contrast/chroma/centroid/tonnetz': ['mfcc', 'spectral_contrast', 'chroma_cens', 'spectral_centroid', 'tonnetz'],
    'mfcc/contrast/chroma/centroid/zcr': ['mfcc', 'spectral_contrast', 'chroma_cens', 'spectral_centroid', 'zcr'],
}
'''
#
# scores, times = test_classifiers_features(classifiers, feature_sets, multi_label=True)



class MachineLearning:
    def __init__(self, *args, **kwargs):
        self.data = None
        self.model = None
        self.score = None
        self.get_params(*args, **kwargs)

    def get_params(self, *args, **kwargs):
        self.data = kwargs.get('data')

    def train(self, *args, **kwargs):
        pass

    @staticmethod
    def training_method(algorithm=None):
        ml_algorithms = {
            'knn': KNN,
            'naivebayes': NaiveBayes,
            'randomforest': RandomForest,
            'decisiontree': DecisionTree,
            'svc': SVC,
            'linearsvc': LinearSVC,
        }
        try:
            return ml_algorithms[algorithm]()
        except:
            return MachineLearning()


class KNN(MachineLearning):
    def train(self, *args, **kwargs):
        self.get_params(*args, **kwargs)
        if self.data is None:
            return []

        # X_train, X_test, y_train, y_test
        X_train = self.data['X_train']
        y_train = self.data['y_train']
        X_test = self.data['X_test']
        y_test = self.data['y_test']

        self.model = skl.neighbors.KNeighborsClassifier(n_neighbors=3)
        self.model.fit(X_train, y_train)
        self.score = self.model.score(X_test, y_test)
        return self.model


class NaiveBayes(MachineLearning):
    def train(self, *args, **kwargs):
        self.get_params(*args, **kwargs)
        if self.data is None:
            return []

        # X_train, X_test, y_train, y_test
        X_train = self.data['X_train']
        y_train = self.data['y_train']
        X_test = self.data['X_test']
        y_test = self.data['y_test']

        self.model = skl.naive_bayes.GaussianNB()
        self.model.fit(X_train, y_train)
        self.score = self.model.score(X_test, y_test)
        return self.model


class RandomForest(MachineLearning):
    def train(self, *args, **kwargs):
        self.get_params(*args, **kwargs)
        if self.data is None:
            return []

        # X_train, X_test, y_train, y_test
        X_train = self.data['X_train']
        y_train = self.data['y_train']
        X_test = self.data['X_test']
        y_test = self.data['y_test']

        self.model = skl.ensemble.RandomForestClassifier()
        self.model.fit(X_train, y_train)
        self.score = self.model.score(X_test, y_test)
        return self.model


class DecisionTree(MachineLearning):
    def train(self, *args, **kwargs):
        self.get_params(*args, **kwargs)
        if self.data is None:
            return []

        # X_train, X_test, y_train, y_test
        X_train = self.data['X_train']
        y_train = self.data['y_train']
        X_test = self.data['X_test']
        y_test = self.data['y_test']

        self.model = skl.tree.DecisionTreeClassifier()
        self.model.fit(X_train, y_train)
        self.score = self.model.score(X_test, y_test)
        return self.model


class SVC(MachineLearning):
    def train(self, *args, **kwargs):
        self.get_params(*args, **kwargs)
        if self.data is None:
            return []

        # X_train, X_test, y_train, y_test
        X_train = self.data['X_train']
        y_train = self.data['y_train']
        X_test = self.data['X_test']
        y_test = self.data['y_test']

        self.model = skl.svm.SVC()
        self.model.fit(X_train, y_train)
        self.score = self.model.score(X_test, y_test)
        return self.model


class LinearSVC(MachineLearning):
    def train(self, *args, **kwargs):
        self.get_params(*args, **kwargs)
        if self.data is None:
            return []

        # X_train, X_test, y_train, y_test
        X_train = self.data['X_train']
        y_train = self.data['y_train']
        X_test = self.data['X_test']
        y_test = self.data['y_test']

        self.model = skl.svm.LinearSVC()
        self.model.fit(X_train, y_train)
        self.score = self.model.score(X_test, y_test)
        return self.model