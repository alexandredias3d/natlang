import nltk
import os

from pt_lacioweb import *
from pt_floresta import *
from pt_mac_morpho import *

class Corpora:
    '''
        Class to manage corpora.
    '''

    def __init__(self, corpora, universal=True):
        '''
            Class constructor. Receive a list of corpora names and a
            boolean to control universal mapping.
        '''
        self.tagged_sents = []
        self._validate_corpus_names(corpora)
        for corpus in corpora:
            self._download_corpus(corpus)
            self.read_corpus(corpus)

    @classmethod
    def _validate_corpus_names(cls, corpora):
        '''
            Map a string name into the available corpus
            class in natlang.
        '''
        mapping = {'MACMORPHO': MacMorpho,
                   'FLORESTA': Floresta,
                   'LACIOWEB': LacioWeb}

        def _clean(name):
            return name.upper().replace(' ', '').replace('_', '')

        try:
            for idx, name in enumerate(corpora):
                corpora[idx] = mapping[_clean(name)]
        except KeyError:
            raise Exception('natlang.postagging.corpus: invalid corpus name')

    @classmethod
    def _download_corpus(cls, corpus):
        '''
            Download the corpus if it has not yet been downloaded.
        '''
        try:
            nltk.data.find(f'corpora/{corpus.name}')
        except LookupError:
            nltk.download(corpus.name, quiet=True)

        # Hardcoding LacioWeb here. TODO: provide better solution.
        if corpus.name == 'lacioweb':
            if not os.path.exists(f'corpus/{corpus.name}'):
                LacioWeb._get_corpus('full')

    def read_corpus(self, cls, universal=True):
        '''
            Read the given corpus and convert it to universal tagset
            if universal is set to True.
        '''
        corpus = cls(universal=universal)
        if universal:

            # Due to LacioWeb corpus being a dict, need this check.
            # TODO: provide better solution.
            if isinstance(corpus, LacioWeb):
                self.tagged_sents += corpus.universal_tagged_sents['full']
            else:
                self.tagged_sents += corpus.universal_tagged_sents
