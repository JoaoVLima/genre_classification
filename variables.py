import os

pwd = os.getcwd()

# Definindo variaveis para o diretorio da base
FMA = '/run/user/1001/gvfs/smb-share:server=10.11.12.28,share=tcc'
small = 'fma_small'
medium = 'fma_medium'
large = 'fma_large'
full = 'fma_full'
metadata = 'fma_metadata'

# Difinindo o diretorio que vamos usar
AUDIO_DIR = f'{FMA}/{full}'
METADATA_DIR = f'{FMA}/{metadata}'
LOG_DIR = f'{FMA}/log'
FEATURE_DIR = f'{FMA}/features'

MODEL_DIR = f'{pwd}/models'

sampling_rate = 44100

