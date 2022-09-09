from multiprocessing import Pool

import librosa as lr
from variables import *
import utils
import pandas as pd
import numpy as np
from ast import literal_eval
import datetime
from tqdm import tqdm
from sklearn.svm import LinearSVC
import pickle

def tratar_campo(campo):
    if not pd.isna(campo):
        campo = literal_eval(campo)
    return campo

def train_models(funcoes_de_feature_extraction, music_ids, funcoes_de_ml, genres, tracks_ids, tracks):
    data_file = pd.read_csv(FEATURE_DIR + '/data.csv', index_col='id')
    data_inicio_file = pd.read_csv(FEATURE_DIR + '/data_start.csv', index_col='id')
    data_meio_file = pd.read_csv(FEATURE_DIR + '/data_middle.csv', index_col='id')
    data_fim_file = pd.read_csv(FEATURE_DIR + '/data_end.csv', index_col='id')

    datas = [
        data_file,
        data_inicio_file,
        data_meio_file,
        data_fim_file,
    ]

    for i, data in enumerate(datas):
        for genre, id in music_ids.items():
            for ml in funcoes_de_ml:
                model = ml()

                features = data.loc[:, [funcao.__name__ for funcao in funcoes_de_feature_extraction]][list(music_ids.values())].tolist()

                results = [1 if i == genre else 0 for i in music_ids]

                model.fit(features, results)

                pickle.dumps(model, open(f'{MODEL_DIR}/{genre}{i}.pkl', 'wb'))

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