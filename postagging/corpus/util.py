"""
    Utility module that provide useful functionality for easy
    management of corpora.
"""
import abc
import os

import nltk

class Corpus(abc.ABC):
    """
        Abstract Corpus class that cannot be instantiated.
        Provide higher level functions for mapping between
        different tagsets.
    """

    @abc.abstractmethod
    def __init__(self, corpus=None, default='X', folder='corpus'):
        """
            Abstract Corpus class constructor.

            :param corpus TaggedCorpusReader: corpus reader from NLTK
            :param default str: default tag to be used if key is not found
            :param mapping dict: dictionary to map from one tagset to another
            :param folder str: corpus folder name
        """
        self._default = default
        self._folder = folder
        self.corpus = corpus
        self.mapping = None
        self.mapped_tagged_sents = []

    def _download_corpus(self):
        """
            Download the corpus if it has not been downloaded yet.
        """
        module = self.corpus.__module__.split('.')[0]
        if 'nltk' in module:
            self._download_from_nltk()
        else:
            pass

    def _download_from_nltk(self):
        """
            Check whether or not the current corpus is already on NLTK
            corpora folder. Download the corpus if it is not available.
        """
        name = self.corpus.root.path.split('/')[-1]
        try:
            nltk.data.find(f'corpora/{name}')
        except LookupError:
            nltk.download(name, quiet=True)

    @abc.abstractmethod
    def _download_from_url(self):
        """
            Check whether or not the corpus folder exists and return a
            boolean to the caller: True if folder was created, meaning
            that the corpus need to be downloaded; and False if folder
            already exists, meaning that the corpus do not need to be
            downloaded.

            :return: True if folder did not exist, False otherwise
        """
        try:
            os.makedirs(self._folder)
            return True
        except FileExistsError:
            return False

    def map_word_tag(self, word, tag):
        """
            Map a single word-tag tuple to the tagset present in
            the mapping dictionary.

            :param word str: current word in sentence
            :param tag str: current PoS tag of the given word
            :return: tuple word-tag' where tag' is the new tag
        """
        return word, self.mapping.get(tag, self._default)

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


#class Corpora:
#    '''
#        Class to manage corpora.
#    '''
#
#    def __init__(self, corpora, universal=True):
#        '''
#            Class constructor. Receive a list of corpora names and a
#            boolean to control universal mapping.
#        '''
#        self.tagged_sents = []
#        self._validate_corpus_names(corpora)
#        for corpus in corpora:
#            self._download_corpus(corpus)
#            self.read_corpus(corpus)
#
#    @classmethod
#    def _validate_corpus_names(cls, corpora):
#        '''
#            Map a string name into the available corpus
#            class in natlang.
#        '''
#        mapping = {'MACMORPHO': MacMorpho,
#                   'FLORESTA': Floresta,
#                   'LACIOWEB': LacioWeb}
#
#        def _clean(name):
#            return name.upper().replace(' ', '').replace('_', '')
#
#        try:
#            for idx, name in enumerate(corpora):
#                corpora[idx] = mapping[_clean(name)]
#        except KeyError:
#            raise Exception('natlang.postagging.corpus: invalid corpus name')
#
#    @classmethod
#    def _download_corpus(cls, corpus):
#        '''
#            Download the corpus if it has not yet been downloaded.
#        '''
#        try:
#            nltk.data.find(f'corpora/{corpus.name}')
#        except LookupError:
#            nltk.download(corpus.name, quiet=True)
#
#        # Hardcoding LacioWeb here. TODO: provide better solution.
#        if corpus.name == 'lacioweb':
#            if not os.path.exists(f'corpus/{corpus.name}'):
#                LacioWeb._get_corpus('full')
#
#    def read_corpus(self, cls, universal=True):
#        '''
#            Read the given corpus and convert it to universal tagset
#            if universal is set to True.
#        '''
#        corpus = cls(universal=universal)
#        if universal:
#
#            # Due to LacioWeb corpus being a dict, need this check.
#            # TODO: provide better solution.
#            if isinstance(corpus, LacioWeb):
#                self.tagged_sents += corpus.universal_tagged_sents['full']
#            else:
#                self.tagged_sents += corpus.universal_tagged_sents
