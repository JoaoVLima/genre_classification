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

import utils
import progress

from variables import *
from feature_extraction import *

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

# NAO EXECUTAR
data_file = open(FEATURE_DIR + '/data.csv', 'w')
data_s_file = open(FEATURE_DIR + '/data_s.csv', 'w')
data_m_file = open(FEATURE_DIR + '/data_m.csv', 'w')
data_e_file = open(FEATURE_DIR + '/data_e.csv', 'w')

data_file.write('id')
data_s_file.write('id')
data_m_file.write('id')
data_e_file.write('id')

for funcao in funcoes_de_feature_extraction:
    data_file.write(f',{funcao.__name__}')
    data_s_file.write(f',{funcao.__name__}')
    data_m_file.write(f',{funcao.__name__}')
    data_e_file.write(f',{funcao.__name__}')

data_file.write('\n')
data_s_file.write('\n')
data_m_file.write('\n')
data_e_file.write('\n')

data_file.close()
data_s_file.close()
data_m_file.close()
data_e_file.close()

current_id = progress.CURRENT_ID
errors_id = progress.ERRORS_ID

tracks_ids =  list(tracks.index)

tracks_ids = tracks_ids[tracks_ids.index(current_id):]


def runbatch(inicio, fim):
    threadlist = []
    for id in tracks_ids[inicio:fim] :
        t = threading.Thread(target=teste, args=(id,))
        threadlist.append(t)
        t.start()

    for tr in threadlist:
        tr.join()
        print("Finished")

def etl(filename: str) -> tuple[str, float]:
    # extract
    start_t = time.perf_counter()
    samplerate, data = scipy.io.wavfile.read(filename)

    # do some transform
    eps = .1
    data += np.random.normal(scale=eps, size=len(data))
    data = np.clip(data, -1.0, 1.0)

    # load (store new form)
    new_filename = filename.removesuffix(".wav") + "-transformed.wav"
    scipy.io.wavfile.write(new_filename, samplerate, data)
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


utils.FfmpegLoader()


def extract_features(data, funcoes:list):

    for funcao in funcoes:
        data_inicio_treino_x = funcao(y=data_inicio, sr=sampling_rate)
        data_meio_treino_x = funcao(y=data_meio, sr=sampling_rate)
        data_fim_treino_x = funcao(y=data_fim, sr=sampling_rate)


    data_file = open(FEATURE_DIR + '/data.csv', 'a')
    data_inicio_file = open(FEATURE_DIR + '/data_inicio.csv', 'a')
    data_meio_file = open(FEATURE_DIR + '/data_meio.csv', 'a')
    data_fim_file = open(FEATURE_DIR + '/data_fim.csv', 'a')
    log_file = open(LOG_DIR + '/log.txt', 'a')

    # Para cada track da base
    for id in tracks_ids:
        try:
            ## Open files
            progress_file = open(LOG_DIR + '/progress.py', 'w')

            ## Salva as variaveis de progresso
            progress_file.write(f'CURRENT_ID = {id}\n')
            progress_file.write(f'ERRORS_ID = {errors_id}\n')
            progress_file.close()

            # Pega o caminho do arquivo de audio
            filename = utils.get_audio_path(AUDIO_DIR, id)

            # Carrega os dados do arquivo
            data, sampling_rate = lr.load(filename, sr=sampling_rate, mono=True)

            # Time decomposition
            data_inicio, data_meio, data_fim = extrair30s(data, sampling_rate)

            data_file.write(f'{id}')
            data_inicio_file.write(f'{id}')
            data_meio_file.write(f'{id}')
            data_fim_file.write(f'{id}')

            # Feature Extraction
            for funcao in funcoes_de_feature_extraction:
                data_treino_x = funcao(y=data, sr=sampling_rate)
                data_inicio_treino_x = funcao(y=data_inicio, sr=sampling_rate)
                data_meio_treino_x = funcao(y=data_meio, sr=sampling_rate)
                data_fim_treino_x = funcao(y=data_fim, sr=sampling_rate)

                # Adiciona as features nos csv
                data_file.write(f',"{str(data_treino_x.tolist())}"')
                data_inicio_file.write(f',"{str(data_inicio_treino_x.tolist())}"')
                data_meio_file.write(f',"{str(data_meio_treino_x.tolist())}"')
                data_fim_file.write(f',"{str(data_fim_treino_x.tolist())}"')

            break
        except Exception as e:
            errors_id.append(id)
            log_file.write(f'{datetime.datetime.now()} | ERROR | id={id}, error={e}\n')
        else:
            log_file.write(f'{datetime.datetime.now()} | SUCCESS | id={id}\n')
        finally:

    log_file.write(f'{datetime.datetime.now()} | FINISH | hopefully...\n')

    data_file.close()
    data_inicio_file.close()
    data_meio_file.close()
    data_fim_file.close()
    log_file.close()


    progress_file.close()

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