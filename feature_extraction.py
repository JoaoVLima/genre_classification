import librosa as lr


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
