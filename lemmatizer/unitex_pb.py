import re
import os
import zipfile

import wget

class UnitexPBDictionary:
    '''
        Dictionary class to handle Unitex-PB dictionaries from NILC.
    '''
    def __init__(self):
        self.root_path = 'dictionary/unitex'
        self.root_url = ('http://www.nilc.icmc.usp.br/nilc/projects/'
                         'unitex-pb/web/files/{}')

        self.unitag = self._universal_mapping()
        self.unitexpb_delaf = {}

    def _download_dictionary(self, name):
        '''
            Download the given Unitex-PB from NILC website.
        '''
        filename = '{}.zip'.format(name)
        wget.download(self.root_url.format(filename),
                      out=f'{self.root_path}/{filename}')

    @classmethod
    def _validate_dictionary_name(cls, name, version=2):
        '''
            Check if the given name is a valid dictionary name.
            Available dictionaries are DELAS, DELAF and DELACF.
            The first two dictionaries have two versions.
        '''
        valid = {'DELAS': {1: 'DELAS_PB', 2: 'DELAS_PB_v2'},
                 'DELAF': {1: 'DELAF_PB', 2: 'DELAF_PB_v2'},
                 'DELACF': {1: 'DELACF_PB', 2: 'DELACF_PB'}}

        error_msg = '''
                        Valid names are DELAS, DELAF, and DELACF.
                    '''
        name = name.upper()
        if name in valid:
            return valid[name][version]
        raise Exception(error_msg)

    def _extract_dictionary(self, ):
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

    def get_dictionary(self, names):
        '''
            Get each dictionary correctly named in names. Available
            options are DELAS, DELAF, DELACF.
        '''
        os.makedirs(self.root_path, exist_ok=True)

        if isinstance(names, list):
            for name in names:
                self._download_dictionary(self._validate_dictionary_name(name))
        else:
            self._download_dictionary(self._validate_dictionary_name(names))

    def _validate_dictionary_filename(self, name):
        '''
        Get the actual filename of the dictionary as it might have a
            different name from the zipfile.
        '''
        for file in os.listdir(f'{self.root_path}'):
            if file.endswith('.dic'):
                if name.upper() in file.upper():
                    return file
        raise Exception(f'Could not find a filename for {name}')

    @classmethod
    def _universal_mapping(cls):
        '''
            Provide a mapping from the UnitexPB tagset used in the
            dictionary. The following tags were directly extracted
            from the data.

            UnitexPB dictionary manual:
            http://www.nilc.icmc.usp.br/nilc/projects/unitex-pb/web/files/Formato_DELAF_PB.pdf
        '''

        unitag = {}

        # Punctuation: .
        # No punctuation in the dictionary

        # Adjectives: ADJ
        unitag.update({k: 'ADJ' for k in ['A']})

        # Numbers: NUM
        unitag.update({k: 'NUM' for k in ['DET+Num']})

        # Adverbs: ADV
        unitag.update({k: 'ADV' for k in ['ADV']})

        # Conjunctions: CONJ
        unitag.update({k: 'CONJ' for k in ['CONJ']})

        # Determiners: DET
        unitag.update({k: 'DET' for k in ['Det+Art+Def', 'Det+Art+Ind',
                                          'DET+Art+Def', 'DET+Art+Ind']})

        # Nouns: NOUN
        # N+Pr seems to be a proper noun: buarque, coimbra...
        unitag.update({k: 'NOUN' for k in ['N', 'N+Pr']})

        # Pronouns: PRON
        unitag.update({k: 'PRON' for k in ['PRO+Dem', 'PRO+Ind', 'PRO+Int',
                                           'PRO+Pes', 'PRO+Pos', 'PRO+Rel',
                                           'PRO+Tra']})

        # Particles: PRT
        unitag.update({k: 'PRT' for k in ['PFX']})

        # Adposition: ADP
        unitag.update({k: 'ADP' for k in ['PREP', 'PREPXADV',
                                          'PREPXDET+Art+Def',
                                          'PREPXDET+Art+Ind',
                                          'PREPXDET+Dem',
                                          'PREPXDET+Ind', 'PREPXPREP',
                                          'PREPXPREP',
                                          'PREPXPRO+Dem', 'PREPXPRO+Int',
                                          'PREPXPRO+Ind', 'PREPXPRO+Pes',
                                          'PREPXPRO+Rel', 'PROXPRO+DemXInd',
                                          'PROXPRO+PosXTra']})

        # Verbs: VERB
        unitag.update({k: 'VERB' for k in ['V', 'V+PRO']})

        # Miscellaneous: X
        unitag.update({k: 'X' for k in ['INTERJ']})

        return unitag

    def read_dictionary(self, name):
        '''
            Read the Unitex-PB dictionary into a Python dictionary.
            Currently only DELAF is supported.
        '''
        filename = self._validate_dictionary_filename(name)
        name = name.upper()
        if name == 'DELAF':
            self._read_delaf(filename)
        elif name == 'DELAS':
            raise NotImplementedError('Currently, DELAS is not supported')
        elif name == 'DELACF':
            raise NotImplementedError('Currently, DELACF is not supported')

    def _read_delaf(self, filename, universal_tagset=True):
        '''
            Read UnitexPB DELAF dictionary.
        '''
        with open(f'{self.root_path}/{filename}', mode='r',
                  encoding='utf-8') as file:
            raw = file.read().replace('\ufeff', '', 1).split('\n')[:-2]

        pattern = re.compile(r'''(?P<word>.*),
                                 (?P<canon>.*)\.
                                 (?P<postag>[a-zA-z+]*)(:|$)
                             ''', re.VERBOSE)
        for entry in raw:
            match = re.match(pattern, entry)
            # Should I create a Lexeme class/namedtuple?
            if universal_tagset:
                self.unitexpb_delaf[(match['word'],
                                     self.unitag[match['postag']])] = match['canon']
            else:
                self.unitexpb_delaf[(match['word'],
                                     match['postag'])] = match['canon']
