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
import machine_learning


from variables import *
from feature_extraction import *

def main():
    # Carregando metadados.
    tracks = utils.load(METADATA_DIR + '/tracks.csv')
    # genres = utils.load(METADATA_DIR + '/genres.csv')

    # Tratando a Base de dados FMA
    # tracks_sem_genero = tracks[('track', 'genre_top')].isna()
    tracks_com_genero = tracks[('track', 'genre_top')].notna()
    tracks = tracks[tracks_com_genero]

    # Limitando a base
    # tracks_subset = tracks['set', 'subset'] <= 'full'
    # tracks = tracks[tracks_subset]

    # Feature Extraction
    funcoes_de_feature_extraction = [
        lr.feature.chroma_stft,
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
        lr.feature.fourier_tempogram
    ]

    funcoes_de_ml = [
        LinearSVC,
        SVC,
        KNeighborsClassifier
    ]

    genres = [
        'Experimental',
        'Electronic',
        'Rock',
        'Instrumental',
        'Pop',
        'Folk',
        'Punk',
        'Avant - Garde',
        'Hip - Hop',
        'Noise',
        'Ambient',
        'Experimental Pop',
        'Electroacoustic',
        'Lo - Fi',
        'Soundtrack',
        'Ambient Electronic',
        'Indie - Rock',
        'International',
        'Improv',
        'Singer - Songwriter',
        'Jazz',
        'Classical',
        'Garage',
        'IDM',
        'Field Recordings',
        'Musique Concrete',
        'Glitch',
        'Drone',
        'Psych - Rock',
        'Loud - Rock',
        'Psych - Folk',
        'Industrial',
        'Chip Music',
        'Techno',
        'Noise - Rock',
        'Downtempo',
        'Country',
        'Post - Rock',
        'Sound Collage',
        'Spoken',
        'Post - Punk',
        'Synth Pop',
        'Blues',
        'Trip - Hop',
        'Free - Jazz',
        'Unclassifiable',
        'Soul - RnB',
        'Metal',
        'House',
        'Hardcore',
        'Dance',
        'Sound Art',
        'Minimalism',
        'Freak - Folk',
        'Contemporary Classical',
        'Chiptune',
        'Hip - Hop Beats',
        'Dubstep',
        'Minimal Electronic',
        'Power - Pop',
        'Americana',
        'Novelty',
        'Reggae - Dub',
        'Old - Time / Historic',
        'Chill - out',
        'Progressive',
        'Audio Collage',
        'Funk',
        'Shoegaze',
        'Free - Folk',
        'Alternative Hip - Hop',
        'Breakbeat',
        'Easy Listening',
        'Europe',
        'Krautrock',
        'Sound Poetry',
        'Rap',
        'Composed Music',
        'Balkan',
        'No Wave',
        'Latin America',
        'Electro - Punk',
        'Radio',
        'New Wave',
        'Breakcore - Hard',
        'Drum & Bass',
        'Spoken Weird',
        'Space - Rock',
        'Goth',
        'Lounge',
        'French',
        'Disco',
        'New Age',
        'Compilation',
        'African',
        'Grindcore',
        'Sound Effects',
        'Poetry',
        '20th Century Classical',
        'Jazz: Out',
        'Holiday',
        'Surf',
        'Jungle',
        'Sludge',
        'Brazilian',
        'Comedy',
        'Choral',
        'Music',
        'Indian',
        'Radio',
        'Art',
        'Abstract',
        'Hip - Hop',
        'Kid - Friendly',
        'Death - Metal',
        'Bigbeat',
        'Bluegrass',
        'Spoken',
        'Word',
        'Middle',
        'East',
        'Thrash',
        'Chamber',
        'Music',
        'British',
        'Folk',
        'Opera',
        'Black - Metal',
        'Rockabilly',
        'Interview',
        'Nu - Jazz',
        'Asia - Far',
        'East',
        'Rock',
        'Opera',
        'Spanish',
        'Latin',
        'Romany(Gypsy)',
        'Afrobeat',
        'Modern',
        'Jazz',
        'Big',
        'Band / Swing',
        'Jazz: Vocal',
        'Reggae - Dancehall',
        'Nerdcore',
        'Country & Western',
        'Skweee',
        'Radio',
        'Theater',
        'Celtic',
        'Christmas',
        'Cumbia',
        'Gospel',
        'Polka',
        'Turkish',
        'Klezmer',
        'Wonky',
        'Flamenco',
        'Easy',
        'Listening: Vocal',
        'North',
        'African',
        'Tango',
        'Fado',
        'Talk',
        'Radio',
        'Symphony',
        'Pacific',
        'Musical',
        'Theater',
        'South',
        'Indian',
        'Traditional',
        'Salsa',
        'Banter',
        'Western',
        'Swing',
        'N.Indian',
        'Traditional',
        'Deep',
        'Funk',
        'Be - Bop',
        'Bollywood',
    ]

    current_id = progress.CURRENT_ID
    errors_id = progress.ERRORS_ID

    tracks_ids = list(tracks.index)
    tracks_ids = tracks_ids[tracks_ids.index(current_id):]

    music_ids = dict(tracks[('track', 'genre_top')])
    music_ids = {v: k for k, v in music_ids.items()}

    feature_extraction.extract_features(funcoes_de_feature_extraction=funcoes_de_feature_extraction, music_ids=music_ids, genres=genres, tracks_ids=tracks_ids, sampling_rate=sampling_rate, current_id=current_id, errors_id=errors_id)

    machine_learning.train_models(funcoes_de_feature_extraction=funcoes_de_feature_extraction, funcoes_de_ml=funcoes_de_ml, music_ids=music_ids, genres=genres, tracks_ids=tracks_ids, tracks=tracks)


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



