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
    
