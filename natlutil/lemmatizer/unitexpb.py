import abc
import json
import re
import os
import zipfile

import wget


class UnitexPBDictionary(abc.ABC):
    '''
        Dictionary class to handle Unitex-PB dictionaries from NILC.
    '''
    @abc.abstractmethod
    def __init__(self, mapping=None):
        self.root_path = 'dictionary/unitex'
        self.root_url = ('http://www.nilc.icmc.usp.br/nilc/projects/'
                         'unitex-pb/web/files/{}')

        if mapping:
            with open(f'{os.path.dirname(os.path.abspath(__file__))}/map/{mapping}') as file:
                self.mapping = json.load(file)
        else:
            self.mapping = None

        self.dict = {}

    def _download_dict(self, name):
        '''
            Download the given Unitex-PB from NILC website.
        '''
        filename = '{}.zip'.format(name)
        wget.download(self.root_url.format(filename),
                      out=f'{self.root_path}/{filename}')

    def _validate_name(self, name, version=2):
        '''
            Check if the given name is a valid dictionary name.
            Available dictionaries are DELAS, DELAF and DELACF.
            The first two dictionaries have two versions.
        '''
        valid = {'DELAS': {1: 'DELAS_PB', 2: 'DELAS_PB_v2'},
                 'DELAF': {1: 'DELAF_PB', 2: 'DELAF_PB_v2'},
                 'DELACF': {1: 'DELACF_PB', 2: 'DELACF_PB'}}

        error_msg = 'Valid names are DELAS, DELAF, and DELACF.'
        name = name.upper()
        try:
            return valid[name][version]
        except KeyError:
            print(error_msg)

    def _extract_dict(self):
        '''
            Extract downloaded dictionaries at the root path.
        '''
        files = [file for file in os.listdir(self.root_path)
                 if file.endswith('.zip')]

        for file in files:
            try:
                with zipfile.ZipFile(f'{self.root_path}/{file}', 'r') as _zip:
                    _zip.extractall(self.root_path)
            except zipfile.BadZipFile:
                print('Zip file might be corrupted.')
            except zipfile.LargeZipFile:
                print('Trying to unzip a large file without ZIP64 enabled.')

    def _get_dict(self, name, version=2):
        '''
            Get each dictionary correctly named in names. Available
            options are DELAS, DELAF, DELACF.
        '''
        os.makedirs(self.root_path, exist_ok=True)
        self._download_dict(self._validate_name(name, version))
        self._extract_dict()

    def _validate_filename(self, name):
        '''
            Get the actual filename of the dictionary as it might have a
            different name from the zipfile.
        '''
        for file in os.listdir(f'{self.root_path}'):
            if file.endswith('.dic'):
                if name.upper() in file.upper():
                    return file
        raise AttributeError(f'Could not find a filename for {name}')

class UnitexPBDELAF(UnitexPBDictionary):

    def __init__(self, mapping=None, version=2):
        super().__init__(mapping)
        self._name = 'DELAF'

        try:
            self._read(self._validate_filename(self._name))
        except AttributeError:
            self._get_dict(self._name, version)
            self._read(self._validate_filename(self._name))

    def __getitem__(self, key):
        return self.dict[key]

    def _read(self, filename):
        '''
            Read UnitexPB DELAF dictionary.
        '''
        with open(f'{self.root_path}/{filename}', mode='r', encoding='utf-8') as file:
            raw = file.read().replace('\ufeff', '', 1).split('\n')[:-2]

        pattern = re.compile(r'''(?P<word>.*),
                                 (?P<canon>.*)\.
                                 (?P<postag>[a-zA-z+]*)(:|$)
                             ''', re.VERBOSE)
        for entry in raw:
            match = re.match(pattern, entry)
            # Should I create a Lexeme class/namedtuple?
            if self.mapping:
                self.dict[(match['word'], self.mapping[match['postag']])] = match['canon']
            else:
                self.dict[(match['word'],
                                     match['postag'])] = match['canon']
