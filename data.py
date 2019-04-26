import json
import os
import tarfile
import wget

br_reviews_url = 'https://dl.dropboxusercontent.com/s/y6u7rl44nqdoznl/br.tar.gz?dl=0'
en_reviews_url = 'https://dl.dropboxusercontent.com/s/wboo4atryaw19bv/en.tar.gz?dl=0'

folder = '../dat'

def make_folder():
    '''
        Create the dat folder one directory above.
    '''
    os.makedirs(folder, exist_ok=True)

def download_data():
    '''
        Download a subset of data to be used during the meetings of the group.
    '''
    wget.download(br_reviews_url, out=f'{folder}/br.tar.gz')
    wget.download(en_reviews_url, out=f'{folder}/en.tar.gz')
        
def extract_data():
    '''
        Extract the compressed .tar.gz.
    '''
    files = [file for file in os.listdir(folder) if file.endswith('.tar.gz')]
    for file in files:
        try:
            with tarfile.open(f'{folder}/{file}') as tar:
                tar.extractall(f'{folder}/')
        except tarfile.ReadError:
            print('\nThis Python wasn\'t compiled with zlib support. Please, '
                  'extract the data manually.')
        
def prepare():
    '''
        Do everything that is needed to prepare the data.
    '''
    make_folder()
    download_data()
    extract_data()

def retrieve(lang, appid, flatten=True):
    '''
        Retrieve reviews from a game in a given a language.
        The result dictionary is flattened by default
    '''
    path = f'{folder}/{lang}/{appid}/review'
    for file in os.listdir(path):
        with open(f'{path}/{file}') as review:
            if flatten:
                yield flatten_dictionary(json.load(review))
            else:
                yield json.load(review)

def flatten_dictionary(dic, pkey='', sep='_'):
    '''
        Flatten nested dictionaries using a separator.
    '''
    items = []
    for key, value in dic.items():
        new_key = (pkey + sep + key) if pkey else key
        if isinstance(value, dict):
            # If value is a dict, it should be flattened.
            items.extend(flatten_dictionary(value, new_key).items())
        else:
            # If it is not a dict, move on with life.
            items.append((new_key, value))

    return dict(items)
