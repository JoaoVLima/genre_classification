from multiprocessing import Pool

import librosa as lr
from variables import *
import utils
import pandas as pd
import numpy as np
from ast import literal_eval
import datetime
from tqdm import tqdm

# [
#     'chroma_stft',
#     'chroma_cqt',
#     'chroma_cens',
#     'melspectrogram',
#     'mfcc',
#     'spectral_centroid',
#     'spectral_bandwidth',
#     'spectral_contrast',
#     'spectral_rolloff',
#     'poly_features',
#     'tonnetz',
#     'tempogram',
#     'fourier_tempogram'
# ]
#
# [
#     lr.feature.chroma_stft,
#     lr.feature.chroma_cqt,
#     lr.feature.chroma_cens,
#     lr.feature.melspectrogram,
#     lr.feature.mfcc,
#     lr.feature.spectral_centroid,
#     lr.feature.spectral_bandwidth,
#     lr.feature.spectral_contrast,
#     lr.feature.spectral_rolloff,
#     lr.feature.poly_features,
#     lr.feature.tonnetz,
#     lr.feature.tempogram,
#     lr.feature.fourier_tempogram
# ]

# feature_extraction = feature.Feature.extraction_method('mfcc')
#
# feature_extraction(audio).extract()


def criar_csvs(funcoes_de_feature_extraction):
    data_files = [
        open(FEATURE_DIR + '/data.csv', 'w'),
        open(FEATURE_DIR + '/data_start.csv', 'w'),
        open(FEATURE_DIR + '/data_middle.csv', 'w'),
        open(FEATURE_DIR + '/data_end.csv', 'w')
    ]

    for data_file in data_files:
        data_file.write('id')

    for funcao in funcoes_de_feature_extraction:
        for data_file in data_files:
            data_file.write(f',{funcao}')

    for data_file in data_files:
        data_file.write('\n')

    for data_file in data_files:
        data_file.close()

# TODO: Caso a musica tenha menos que 1min 30sec, dividir em 3 partes.
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


def extract_features(tracks_ids, sampling_rate, music_ids, genres, current_id, errors_id, fe_functions):
    data_file = pd.read_csv(FEATURE_DIR + '/data.csv', index_col='id')
    data_inicio_file = pd.read_csv(FEATURE_DIR + '/data_start.csv', index_col='id')
    data_meio_file = pd.read_csv(FEATURE_DIR + '/data_middle.csv', index_col='id')
    data_fim_file = pd.read_csv(FEATURE_DIR + '/data_end.csv', index_col='id')
    log_file = open(LOG_DIR + '/log.txt', 'a')

    data_file = data_file.astype(object)
    data_inicio_file = data_inicio_file.astype(object)
    data_meio_file = data_meio_file.astype(object)
    data_fim_file = data_fim_file.astype(object)

    # TODO: Adicionar Threads/Paralelismo nessa etapa
    # Para cada track da base
    for id in tqdm(tracks_ids):
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

            # Feature Extraction
            for funcao in fe_functions:
                feature = Feature.extraction_method(funcao)
                data_treino_x = feature.extract(audio=data, sampling_rate=sampling_rate)
                data_inicio_treino_x = feature.extract(audio=data_inicio, sampling_rate=sampling_rate)
                data_meio_treino_x = feature.extract(audio=data_meio, sampling_rate=sampling_rate)
                data_fim_treino_x = feature.extract(audio=data_fim, sampling_rate=sampling_rate)

                # Adiciona as features nos csv
                data_file.loc[:, funcao][id] = data_treino_x.tolist()
                data_inicio_file.loc[:, funcao][id] = data_inicio_treino_x.tolist()
                data_meio_file.loc[:, funcao][id] = data_meio_treino_x.tolist()
                data_fim_file.loc[:, funcao][id] = data_fim_treino_x.tolist()

            data_file.to_csv(FEATURE_DIR + '/data.csv', encoding='utf-8')
            data_inicio_file.to_csv(FEATURE_DIR + '/data_start.csv', encoding='utf-8')
            data_meio_file.to_csv(FEATURE_DIR + '/data_middle.csv', encoding='utf-8')
            data_fim_file.to_csv(FEATURE_DIR + '/data_end.csv', encoding='utf-8')

        except Exception as e:
            errors_id.append(id)
            log_file.write(f'{datetime.datetime.now()} | ERROR | id={id}, error={e}\n')


class Feature:
    def __init__(self, *args, **kwargs):
        self.data = None
        self.audio = None
        self.sampling_rate = None
        self.get_params(*args, **kwargs)

    def get_params(self, *args, **kwargs):
        self.audio = kwargs.get('audio')
        self.sampling_rate = kwargs.get('sampling_rate')

    def extract(self, *args, **kwargs):
        pass

    @staticmethod
    def extraction_method(feature=None):
        features = {
            'chroma_stft': ChromaStft(),
            'chroma_cqt': ChromaCqt(),
            'chroma_cens': ChromaCens(),
            'melspectrogram': Melspectrogram(),
            'mfcc': MFCC(),
            'spectral_centroid': SpectralCentroid(),
            'spectral_bandwidth': SpectralBandwidth(),
            'spectral_contrast': SpectralContrast(),
            'spectral_rolloff': SpectralRolloff(),
            'poly_features': PolyFeatures(),
            'tonnetz': Tonnetz(),
            'tempogram': Tempogram(),
            'fourier_tempogram': FourierTempogram(),
        }
        try:
            return features[feature]
        except:
            return Feature()


class ChromaStft(Feature):
    def extract(self, *args, **kwargs):
        self.get_params(*args, **kwargs)
        if self.audio is None:
            return []

        self.data = lr.feature.chroma_stft(y=self.audio, sr=self.sampling_rate)
        return self.data


class ChromaCqt(Feature):
    def extract(self, *args, **kwargs):
        self.get_params(*args, **kwargs)
        if self.audio is None:
            return []

        self.data = lr.feature.chroma_cqt(y=self.audio, sr=self.sampling_rate)
        return self.data


class ChromaCens(Feature):
    def extract(self, *args, **kwargs):
        self.get_params(*args, **kwargs)
        if self.audio is None:
            return []

        self.data = lr.feature.chroma_cens(y=self.audio, sr=self.sampling_rate)
        return self.data


class Melspectrogram(Feature):
    def extract(self, *args, **kwargs):
        self.get_params(*args, **kwargs)
        if self.audio is None:
            return []

        self.data = lr.feature.melspectrogram(y=self.audio, sr=self.sampling_rate)
        return self.data


class MFCC(Feature):
    def extract(self, *args, **kwargs):
        self.get_params(*args, **kwargs)
        if self.audio is None:
            return []

        self.data = lr.feature.mfcc(y=self.audio, sr=self.sampling_rate)
        return self.data


class SpectralCentroid(Feature):
    def extract(self, *args, **kwargs):
        self.get_params(*args, **kwargs)
        if self.audio is None:
            return []

        self.data = lr.feature.spectral_centroid(y=self.audio, sr=self.sampling_rate)
        return self.data


class SpectralBandwidth(Feature):
    def extract(self, *args, **kwargs):
        self.get_params(*args, **kwargs)
        if self.audio is None:
            return []

        self.data = lr.feature.spectral_bandwidth(y=self.audio, sr=self.sampling_rate)
        return self.data


class SpectralContrast(Feature):
    def extract(self, *args, **kwargs):
        self.get_params(*args, **kwargs)
        if self.audio is None:
            return []

        self.data = lr.feature.spectral_contrast(y=self.audio, sr=self.sampling_rate)
        return self.data


class SpectralRolloff(Feature):
    def extract(self, *args, **kwargs):
        self.get_params(*args, **kwargs)
        if self.audio is None:
            return []

        self.data = lr.feature.spectral_rolloff(y=self.audio, sr=self.sampling_rate)
        return self.data


class PolyFeatures(Feature):
    def extract(self, *args, **kwargs):
        self.get_params(*args, **kwargs)
        if self.audio is None:
            return []

        self.data = lr.feature.poly_features(y=self.audio, sr=self.sampling_rate)
        return self.data


class Tonnetz(Feature):
    def extract(self, *args, **kwargs):
        self.get_params(*args, **kwargs)
        if self.audio is None:
            return []

        self.data = lr.feature.tonnetz(y=self.audio, sr=self.sampling_rate)
        return self.data


class Tempogram(Feature):
    def extract(self, *args, **kwargs):
        self.get_params(*args, **kwargs)
        if self.audio is None:
            return []

        self.data = lr.feature.tempogram(y=self.audio, sr=self.sampling_rate)
        return self.data


class FourierTempogram(Feature):
    def extract(self, *args, **kwargs):
        self.get_params(*args, **kwargs)
        if self.audio is None:
            return []

        self.data = lr.feature.fourier_tempogram(y=self.audio, sr=self.sampling_rate)
        return self.data
