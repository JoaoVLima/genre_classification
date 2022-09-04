import os
import IPython.display as ipd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
import librosa as lr
import datetime
import json
import threading

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


def extract_features(data, funcoes):

    if type(funcoes)==list:
        for funcao in funcoes:
            data_inicio_treino_x = funcao(y=data_inicio, sr=sampling_rate)
            data_meio_treino_x = funcao(y=data_meio, sr=sampling_rate)
            data_fim_treino_x = funcao(y=data_fim, sr=sampling_rate)

    else:
        data_inicio_treino_x = funcoes(y=data_inicio, sr=sampling_rate)
        data_meio_treino_x = funcoes(y=data_meio, sr=sampling_rate)
        data_fim_treino_x = funcoes(y=data_fim, sr=sampling_rate)


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

        print('Ja tem todos os datas')
        print('iniciando feature extraction')
        # Feature Extraction
        for funcao in funcoes_de_feature_extraction:
            print(funcao.__name__)
            data_treino_x = funcao(y=data, sr=sampling_rate)
            print('data')
            print(data_treino_x)
            data_inicio_treino_x = funcao(y=data_inicio, sr=sampling_rate)
            print('data i')
            print(data_inicio_treino_x)
            data_meio_treino_x = funcao(y=data_meio, sr=sampling_rate)
            print('data m')
            print(data_meio_treino_x)
            data_fim_treino_x = funcao(y=data_fim, sr=sampling_rate)
            print('data f')
            print(data_fim_treino_x)

            # Adiciona as features nos csv
            data_file.write(f',"{str(data_treino_x.tolist())}"')
            data_inicio_file.write(f',"{str(data_inicio_treino_x.tolist())}"')
            data_meio_file.write(f',"{str(data_meio_treino_x.tolist())}"')
            data_fim_file.write(f',"{str(data_fim_treino_x.tolist())}"')
            print('escreveu nos csvs')

        print('acabou')
        break
    except Exception as e:
        errors_id.append(id)
        log_file.write(f'{datetime.datetime.now()} | ERROR | id={id}, error={e}\n')
    else:
        print('deuboa')
        log_file.write(f'{datetime.datetime.now()} | SUCCESS | id={id}\n')
    finally:

print('final')
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

data_inicio_treino_x

tracks['set', 'subset']


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

## Testando o modelo