"""
    Utility module that provide an abstract class to handle
    a corpus.
"""
import abc
import os

import nltk
import wget


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
        self._url = ''
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
            self._download_from_url()

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

    def _download_from_url(self):
        """
            Try to create the folder to save the corpus. If the folder
            already exists, there is no need to download the corpus.
            Otherwise, call wget on the url.
        """
        try:
            os.makedirs(self._folder)
            wget.download(self._url, out=self._folder)
        except FileExistsError:
            pass

    def _to_universal(self):
        """
            Convert the tags in the corpus to the universal tagset.
        """
        self.mapping = self._universal_mapping()
        self.mapped_tagged_sents = self.map_corpus_tags()

    @staticmethod
    @abc.abstractmethod
    def _universal_mapping():
        """
            Method to provide the mapping between the original tagset
            and the universal tagset that should be provided by the
            user of a corpus.
        """

    def map_word_tag(self, word_tag):
        """
            Map a single word-tag tuple to the tagset present in
            the mapping dictionary.

            :param word str: current word in sentence
            :param tag str: current PoS tag of the given word
            :return: tuple word-tag' where tag' is the new tag
        """
        word = word_tag[0]
        tag = word_tag[1]
        try:
            return word, self.mapping[tag]
        except KeyError:
            return word, self._default

    def map_sentence_tags(self, sentence):
        """
            Map tags from a sentence to the tagset present in the
            mapping dictionary.

            :param sentence list: list of word-tag tuples
            :return: list of word-tag tuples, where the tags have
                been mapped to the tagset in mapping
        """
        return list(map(self.map_word_tag, sentence))

    def map_corpus_tags(self):
        """
            Map the entire corpus to the tagset present in the
            mapping dictionary.

            :return: entire corpus mapped to the tagset present in
                mapping
        """
        return map(self.map_sentence_tags, self.corpus.tagged_sents())
