import getpass
import os
import sys

__USERNAME = getpass.getuser()

_BASE_DIR = f'/srv/local/data/{__USERNAME}/'

LLAMA_PATH = f'{_BASE_DIR}/LLM_weights/LLAMA/'
MISTRAL_PATH = f'{_BASE_DIR}/LLM_weights/MISTRAL/'

DATA_FOLDER = os.path.join(_BASE_DIR, 'NLGUQ')
GENERATION_FOLDER = os.path.join(DATA_FOLDER, 'output')
os.makedirs(GENERATION_FOLDER, exist_ok=True)

# After running pipeline/generate.py, update the following paths to the generated files if necessary.
GEN_PATHS = {
    'coqa': {
        'llama-13b': f'{GENERATION_FOLDER}/llama-13b-hf_coqa_10/0.pkl', #llama-2-13B
        'mistral-7b': f'{GENERATION_FOLDER}/mistral-7b-hf_coqa_10/0.pkl', # Mistral-7B-Instruct-v0.2
   },
   'triviaqa': {
        'llama-13b': f'{GENERATION_FOLDER}/llama-13b-hf_triviaqa_0/0.pkl',
        'mistral-7b': f'{GENERATION_FOLDER}/mistral-7b-hf_triviaqa_10/0.pkl',  
   }
}
