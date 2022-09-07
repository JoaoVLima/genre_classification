import librosa as lr
from variables import *
import utils
import pandas as pd
import datetime


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
            data_file.write(f',{funcao.__name__}')

    for data_file in data_files:
        data_file.write('\n')

    for data_file in data_files:
        data_file.close()


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


def extract_features(tracks_ids, sampling_rate, current_id, errors_id, funcoes_de_feature_extraction):

    data_file = pd.read_csv(FEATURE_DIR + '/data.csv', index_col='id')
    data_inicio_file = pd.read_csv(FEATURE_DIR + '/data_start.csv', index_col='id')
    data_meio_file = pd.read_csv(FEATURE_DIR + '/data_middle.csv', index_col='id')
    data_fim_file = pd.read_csv(FEATURE_DIR + '/data_end.csv', index_col='id')
    log_file = open(LOG_DIR + '/log.txt', 'a')

    data_file = data_file.astype(object)
    data_inicio_file = data_inicio_file.astype(object)
    data_meio_file = data_meio_file.astype(object)
    data_fim_file = data_fim_file.astype(object)

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

            # Feature Extraction
            for funcao in funcoes_de_feature_extraction:
                data_treino_x = funcao(y=data, sr=sampling_rate)
                data_inicio_treino_x = funcao(y=data_inicio, sr=sampling_rate)
                data_meio_treino_x = funcao(y=data_meio, sr=sampling_rate)
                data_fim_treino_x = funcao(y=data_fim, sr=sampling_rate)

                # Adiciona as features nos csv
                data_file.iloc[id].loc[funcao.__name__] = data_treino_x
                data_inicio_file.iloc[id].loc[funcao.__name__] = data_inicio_treino_x
                data_meio_file.iloc[id].loc[funcao.__name__] = data_meio_treino_x
                data_fim_file.iloc[id].loc[funcao.__name__] = data_fim_treino_x

            data_file.to_csv(FEATURE_DIR + '/data.csv', encoding='utf-8')
            data_inicio_file.to_csv(FEATURE_DIR + '/data_start.csv', encoding='utf-8')
            data_meio_file.to_csv(FEATURE_DIR + '/data_middle.csv', encoding='utf-8')
            data_fim_file.to_csv(FEATURE_DIR + '/data_end.csv', encoding='utf-8')

            break
        except Exception as e:
            errors_id.append(id)
            log_file.write(f'{datetime.datetime.now()} | ERROR | id={id}, error={e}\n')

    data_file.close()
    data_inicio_file.close()
    data_meio_file.close()
    data_fim_file.close()
    log_file.close()
