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

sampling_rate = 44100

