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

import drive.Shareddrives.tcc.utils as utils
import drive.Shareddrives.tcc.log.progress as progress

def load(filepath):

    filename = os.path.basename(filepath)

    if 'features' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'echonest' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'genres' in filename:
        return pd.read_csv(filepath, index_col=0)

    if 'tracks' in filename:
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

        COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                   ('track', 'genres'), ('track', 'genres_all')]
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

        COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                   ('album', 'date_created'), ('album', 'date_released'),
                   ('artist', 'date_created'), ('artist', 'active_year_begin'),
                   ('artist', 'active_year_end')]
        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])

        SUBSETS = ('small', 'medium', 'large')
        try:
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                    'category', categories=SUBSETS, ordered=True)
        except (ValueError, TypeError):
            # the categories and ordered arguments were removed in pandas 0.25
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                     pd.CategoricalDtype(categories=SUBSETS, ordered=True))

        COLUMNS = [('track', 'genre_top'), ('track', 'license'),
                   ('album', 'type'), ('album', 'information'),
                   ('artist', 'bio')]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype('category')

        return tracks


def get_audio_path(audio_dir, track_id):
    """
    Return the path to the mp3 given the directory where the audio is stored
    and the track ID.

    Examples
    --------
    # >>> import utils
    # >>> AUDIO_DIR = os.environ.get('AUDIO_DIR')
    # >>> utils.get_audio_path(AUDIO_DIR, 2)
    '../data/fma_small/000/000002.mp3'

    """
    tid_str = '{:06d}'.format(track_id)
    return os.path.join(audio_dir, tid_str[:3], tid_str + '.mp3')



# Definindo variaveis para o diretorio da base
FMA = '/run/user/1001/gvfs/smb-share:server=10.11.12.28,share=tcc'
feature = 'feature'
log = 'log'
metadata = 'fma_metadata'
small = 'fma_small'
medium = 'fma_medium'
large = 'fma_large'
full = 'fma_full'

# Difinindo o diretorio que vamos usar
LOG_DIR = f'{FMA+log}'
METADATA_DIR = f'{FMA+metadata}'
AUDIO_DIR = f'{FMA+full}'
FEATURE_DIR = f'{FMA+feature}'

# Carregando metadados.
tracks = utils.load(METADATA_DIR + '/tracks.csv')
genres = utils.load(METADATA_DIR + '/genres.csv')
features = utils.load(METADATA_DIR + '/features.csv')
echonest = utils.load(METADATA_DIR + '/echonest.csv')


## Tratando a Base de dados FMA
tracks_sem_genero = tracks[('track', 'genre_top')].isna()

tracks_com_genero = tracks[('track', 'genre_top')].notna()

tracks = tracks[tracks_com_genero]

# Limitando a base
tipo_tracks = tracks['set', 'subset'] <= 'small'

tracks = tracks[tipo_tracks]


## Feature Extraction
sampling_rate = 44100


def extrair30s(data, sampling_rate):
    segundos = int(data.size / sampling_rate)

    inicio_m = int(segundos / 2) - 15
    fim_m = int(segundos / 2) + 15

    inicio_f = segundos - 35
    fim_f = segundos - 5

    data_inicio = data[0 * sampling_rate:30 * sampling_rate]
    data_meio = data[inicio_m * sampling_rate:fim_m * sampling_rate]
    data_fim = data[inicio_f * sampling_rate:fim_f * sampling_rate]

    return data_inicio, data_meio, data_fim

funcoes_de_feature_extraction = [lr.feature.chroma_stft,
                                 lr.feature.chroma_cqt,
                                 lr.feature.chroma_cens,
                                 lr.feature.melspectrogram,
                                 lr.feature.mfcc,
                                 lr.feature.spectral_centroid,
                                 lr.feature.spectral_bandwidth,
                                 lr.feature.spectral_contrast,
                                 lr.feature.spectral_rolloff,
                                 lr.feature.poly_features,
                                 lr.feature.tonnetz,
                                 lr.feature.tempogram,
                                 lr.feature.fourier_tempogram]

# NAO EXECUTAR
data_file = open(FEATURE_DIR + '/data.csv', 'w')
data_inicio_file = open(FEATURE_DIR + '/data_inicio.csv', 'w')
data_meio_file = open(FEATURE_DIR + '/data_meio.csv', 'w')
data_fim_file = open(FEATURE_DIR + '/data_fim.csv', 'w')

data_file.write('id')
data_inicio_file.write('id')
data_meio_file.write('id')
data_fim_file.write('id')

for funcao in funcoes_de_feature_extraction:
    data_file.write(f',{funcao.__name__}')
    data_inicio_file.write(f',{funcao.__name__}')
    data_meio_file.write(f',{funcao.__name__}')
    data_fim_file.write(f',{funcao.__name__}')

data_file.write('\n')
data_inicio_file.write('\n')
data_meio_file.write('\n')
data_fim_file.write('\n')

data_file.close()
data_inicio_file.close()
data_meio_file.close()
data_fim_file.close()

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