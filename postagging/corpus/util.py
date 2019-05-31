import abc
import nltk
import os

from pt_lacioweb import *
from pt_floresta import *
from pt_mac_morpho import *

class Corpus(abc.ABC):

    @abstractmethod
    def __init__(self, corpus=None, default='X', mapping={}):
        """
            Abstract Corpus class constructor.

            :param corpus TaggedCorpusReader: corpus reader from NLTK
            :param default str: default tag to be used if key is not found
            :param mapping dict: dictionary to map from one tagset to another
        """
        self.corpus = corpus
        self.default = default
        self.mapping = mapping

    def map_word_tag(self, word, tag):
        """
            Map a single word-tag tuple to the tagset present in
            the mapping dictionary.

            :param word str: current word in sentence
            :param tag str: current PoS tag of the given word
            :return: tuple word-tag' where tag' is the new tag
        """
        return word, self.mapping.get()

    def map_sentence_tags(self, sentence):
        """
            Map tags from a sentence to the tagset present in the
            mapping dictionary.

            :param sentence list: list of word-tag tuples
            :return: list of word-tag tuples, where the tags have
                been mapped to the tagset in mapping
        """
        return [self.map_word_tag(word, tag) for word, tag in sentence]

    def map_corpus_tags(self):
        """
            Map the entire corpus to the tagset present in the
            mapping dictionary.

            :return: entire corpus mapped to the tagset present in
                mapping
        """
        return [self.map_sentence_tags(sentence)
                for sentence in self.corpus.tagged_sents()]

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
