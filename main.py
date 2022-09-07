import os
import IPython.display as ipd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa as lr
import datetime
import json
import threading
import time
from multiprocessing import Pool

from tqdm.notebook import tqdm as tqdm_notebook

import keras
from keras.layers import Activation, Dense, Conv1D, Conv2D, MaxPooling1D, Flatten, Reshape

import sklearn as skl
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
from sklearn.utils import shuffle
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, LabelBinarizer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.multiclass import OneVsRestClassifier

import feature_extraction
import utils
import progress

from variables import *
from feature_extraction import *

def main():
    # Carregando metadados.
    tracks = utils.load(METADATA_DIR + '/tracks.csv')
    genres = utils.load(METADATA_DIR + '/genres.csv')

    # Tratando a Base de dados FMA
    tracks_sem_genero = tracks[('track', 'genre_top')].isna()
    tracks_com_genero = tracks[('track', 'genre_top')].notna()
    tracks = tracks[tracks_com_genero]

    # Limitando a base
    tipo_tracks = tracks['set', 'subset'] <= 'small'
    tracks = tracks[tipo_tracks]

    # Feature Extraction
    funcoes_de_feature_extraction = [
        # lr.feature.chroma_stft,
        # lr.feature.chroma_cqt,
        # lr.feature.chroma_cens,
        # lr.feature.melspectrogram,
        lr.feature.mfcc,
        # lr.feature.spectral_centroid,
        # lr.feature.spectral_bandwidth,
        # lr.feature.spectral_contrast,
        # lr.feature.spectral_rolloff,
        # lr.feature.poly_features,
        # lr.feature.tonnetz,
        # lr.feature.tempogram,
        # lr.feature.fourier_tempogram
    ]

    current_id = progress.CURRENT_ID
    errors_id = progress.ERRORS_ID

    tracks_ids = list(tracks.index)

    tracks_ids = tracks_ids[tracks_ids.index(current_id):]

    feature_extraction.extract_features(tracks_ids=tracks_ids, sampling_rate=sampling_rate, current_id=current_id, errors_id=errors_id, funcoes_de_feature_extraction=funcoes_de_feature_extraction)

if __name__ == "__main__":
    main()


def etl(filename):
    # extract
    start_t = time.perf_counter()

    end_t = time.perf_counter()
    return filename, end_t - start_t

def etl_demo():
    filenames = [f"sounds/example{n}.wav" for n in range(24)]
    start_t = time.perf_counter()

    print("starting etl")
    with Pool() as pool:
        results = pool.map(etl, filenames)

        for filename, duration in results:
            print(f"{filename} completed in {duration:.2f}s")

    end_t = time.perf_counter()
    total_duration = end_t - start_t
    print(f"etl took {total_duration:.2f}s total")


'''
funcao = lr.feature.mfcc

data_treino_x = funcao(y=data, sr=sampling_rate)

data_inicio_treino_x = funcao(y=data_inicio, sr=sampling_rate)

data_meio_treino_x = funcao(y=data_meio, sr=sampling_rate)

data_fim_treino_x = funcao(y=data_fim, sr=sampling_rate)

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

def test_classifiers_features(classifiers, feature_sets, multi_label=False):
    columns = list(classifiers.keys()).insert(0, 'dim')
    scores = pd.DataFrame(columns=columns, index=feature_sets.keys())
    times = pd.DataFrame(columns=classifiers.keys(), index=feature_sets.keys())
    for fset_name, fset in tqdm_notebook(feature_sets.items(), desc='features'):
        y_train, y_val, y_test, X_train, X_val, X_test = pre_process(tracks, features_all, fset, multi_label)
        scores.loc[fset_name, 'dim'] = X_train.shape[1]
        for clf_name, clf in classifiers.items():  # tqdm_notebook(classifiers.items(), desc='classifiers', leave=False):
            t = time.process_time()
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            scores.loc[fset_name, clf_name] = score
            times.loc[fset_name, clf_name] = time.process_time() - t
    return scores, times

def format_scores(scores):
    def highlight(s):
        is_max = s == max(s[1:])
        return ['background-color: yellow' if v else '' for v in is_max]
    scores = scores.style.apply(highlight, axis=1)
    return scores.format('{:.2%}', subset=pd.IndexSlice[:, scores.columns[1]:])

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

scores, times = test_classifiers_features(classifiers, feature_sets, multi_label=True)

ipd.display(format_scores(scores))
ipd.display(times.style.format('{:.4f}'))
'''