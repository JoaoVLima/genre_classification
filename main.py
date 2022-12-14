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

# import keras
# from keras.layers import Activation, Dense, Conv1D, Conv2D, MaxPooling1D, Flatten, Reshape

# import feature
import feature
import utils
import progress
import machine_learning

from variables import *


# TODO: Deixar o codigo no seguinte padrão:
#   1 - Codigo em Ingles
#   2 - Comentario em Portugues
#   3 - Imports sem usar o from, a não ser que prejudique a performance
#   4 - Atividades a serem feitas devem estar como to_do no codigo
#   5 - Commits curtos e explicativos
#   6 -

def main(task):
    # Carregando metadados
    tracks = utils.load(METADATA_DIR + '/tracks.csv')
    genres = utils.load(METADATA_DIR + '/genres.csv')
    genres['title'] = genres['title'].apply(utils.tratar_nome_genero)

    # Limitando a base
    # tracks_subset = tracks['set', 'subset'] <= 'full'
    # tracks = tracks[tracks_subset]

    # Feature Extraction
    fe_functions = [
        'chroma_stft',
        'chroma_cqt',
        'chroma_cens',
        'melspectrogram',
        'mfcc',
        'spectral_centroid',
        'spectral_bandwidth',
        'spectral_contrast',
        'spectral_rolloff',
        'poly_features',
        'tonnetz',
        'tempogram',
        'fourier_tempogram',
    ]

    # Machine Learning
    ml_functions = [
        'knn',
        'naivebayes',
        # 'randomforest',
        # 'decisiontree',
        # 'svc',
        # 'linearsvc',
    ]

    # genres = [
    #     'Experimental',
    #     'Electronic',
    #     'Rock',
    #     'Instrumental',
    #     'Pop',
    #     'Folk',
    #     'Punk',
    #     'Avant Garde',
    #     'Hip Hop',
    #     'Noise',
    #     'Ambient',
    #     'Experimental Pop',
    #     'Electroacoustic',
    #     'Lo Fi',
    #     'Soundtrack',
    #     'Ambient Electronic',
    #     'Indie Rock',
    #     'International',
    #     'Improv',
    #     'Singer Songwriter',
    #     'Jazz',
    #     'Classical',
    #     'Garage',
    #     'IDM',
    #     'Field Recordings',
    #     'Musique Concrete',
    #     'Glitch',
    #     'Drone',
    #     'Psych Rock',
    #     'Loud Rock',
    #     'Psych Folk',
    #     'Industrial',
    #     'Chip Music',
    #     'Techno',
    #     'Noise Rock',
    #     'Downtempo',
    #     'Country',
    #     'Post Rock',
    #     'Sound Collage',
    #     'Spoken',
    #     'Post Punk',
    #     'Synth Pop',
    #     'Blues',
    #     'Trip Hop',
    #     'Free Jazz',
    #     'Unclassifiable',
    #     'Soul RnB',
    #     'Metal',
    #     'House',
    #     'Hardcore',
    #     'Dance',
    #     'Sound Art',
    #     'Minimalism',
    #     'Freak Folk',
    #     'Contemporary Classical',
    #     'Chiptune',
    #     'Hip Hop Beats',
    #     'Dubstep',
    #     'Minimal Electronic',
    #     'Power Pop',
    #     'Americana',
    #     'Novelty',
    #     'Reggae Dub',
    #     'Old Time Historic',
    #     'Chill out',
    #     'Progressive',
    #     'Audio Collage',
    #     'Funk',
    #     'Shoegaze',
    #     'Free Folk',
    #     'Alternative Hip Hop',
    #     'Breakbeat',
    #     'Easy Listening',
    #     'Europe',
    #     'Krautrock',
    #     'Sound Poetry',
    #     'Rap',
    #     'Composed Music',
    #     'Balkan',
    #     'No Wave',
    #     'Latin America',
    #     'Electro Punk',
    #     'Radio',
    #     'New Wave',
    #     'Breakcore Hard',
    #     'Drum Bass',
    #     'Spoken Weird',
    #     'Space Rock',
    #     'Goth',
    #     'Lounge',
    #     'French',
    #     'Disco',
    #     'New Age',
    #     'Compilation',
    #     'African',
    #     'Grindcore',
    #     'Sound Effects',
    #     'Poetry',
    #     '20th Century Classical',
    #     'Jazz Out',
    #     'Holiday',
    #     'Surf',
    #     'Jungle',
    #     'Sludge',
    #     'Brazilian',
    #     'Comedy',
    #     'Choral',
    #     'Music',
    #     'Indian',
    #     'Art',
    #     'Abstract',
    #     'Kid Friendly',
    #     'Death Metal',
    #     'Bigbeat',
    #     'Bluegrass',
    #     'Word',
    #     'Middle',
    #     'East',
    #     'Thrash',
    #     'Chamber',
    #     'British',
    #     'Opera',
    #     'Black Metal',
    #     'Rockabilly',
    #     'Interview',
    #     'Nu Jazz',
    #     'Asia Far',
    #     'Spanish',
    #     'Latin',
    #     'Romany Gypsy',
    #     'Afrobeat',
    #     'Modern',
    #     'Big',
    #     'Band Swing',
    #     'Jazz Vocal',
    #     'Reggae Dancehall',
    #     'Nerdcore',
    #     'Country Western',
    #     'Skweee',
    #     'Theater',
    #     'Celtic',
    #     'Christmas',
    #     'Cumbia',
    #     'Gospel',
    #     'Polka',
    #     'Turkish',
    #     'Klezmer',
    #     'Wonky',
    #     'Flamenco',
    #     'Easy',
    #     'Listening Vocal',
    #     'North',
    #     'Tango',
    #     'Fado',
    #     'Talk',
    #     'Symphony',
    #     'Pacific',
    #     'Musical',
    #     'South',
    #     'Traditional',
    #     'Salsa',
    #     'Banter',
    #     'Western',
    #     'Swing',
    #     'N Indian',
    #     'Deep',
    #     'Be Bop',
    #     'Bollywood'
    # ]

    current_id = progress.CURRENT_ID
    errors_id = progress.ERRORS_ID

    tracks_ids = list(tracks.index)
    tracks_ids = tracks_ids[tracks_ids.index(current_id):]

    # music_ids = dict(tracks[('track', 'genre_top')])
    # music_ids = {v: k for k, v in music_ids.items()}.values()

    if task == 'extract_features':
        feature.extract_features(fe_functions=fe_functions,
                                 genres=genres,
                                 tracks_ids=tracks_ids,
                                 sampling_rate=sampling_rate,
                                 current_id=current_id,
                                 errors_id=errors_id)
    elif task == 'train_models':
        machine_learning.train_models(fe_functions=fe_functions,
                                      funcoes_de_ml=ml_functions,
                                      genres=genres,
                                      tracks_ids=tracks_ids,
                                      tracks=tracks)


# if __name__ == "__main__":
#     main()


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
        results = pool.imap_unordered(etl, filenames)

        for filename, duration in results:
            print(f"{filename} completed in {duration:.2f}s")

    end_t = time.perf_counter()
    total_duration = end_t - start_t
    print(f"etl took {total_duration:.2f}s total")
