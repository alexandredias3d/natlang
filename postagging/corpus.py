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
    def __init__(self, corpus=None, default_tag=None, folder=None, url=None,
                 universal=None):
        """
            Abstract Corpus class constructor.

            :param corpus TaggedCorpusReader: corpus reader from NLTK
            :param default str: default tag to be used if key is not found
            :param mapping dict: dictionary to map from one tagset to another
            :param folder str: corpus folder name
        """
        self.default_tag = default_tag
        self.folder = folder
        self.url = url
        self.corpus = corpus
        self.mapping = None
        self.mapped_tagged_sents = []

        self._download_corpus()
        self._read_corpus()

        if universal:
            self._to_universal(f'{self.__class__.__name__}_Universal.txt')

    @property
    def default_tag(self):
        """
            Default tag getter.
        """
        return self._default_tag

    @default_tag.setter
    def default_tag(self, default_tag):
        """
            Default tag setter. Set the default PoS tag to be used whenever an
            unknown tag is found.
        """
        self._default_tag = default_tag if default_tag else 'X'

    @property
    def folder(self):
        """
            Folder getter.
        """
        return self._folder

    @folder.setter
    def folder(self, folder):
        """
            Folder setter. Set the folder where the corpus should be downloaded
            to. This option is used only when the corpus is not available on
            NLTK. Thus, there is the need to manually download it.
        """
        self._folder = folder if folder else 'corpus'

    @property
    def url(self):
        """
            URL getter.
        """
        return self._url

    @url.setter
    def url(self, url):
        """
            URL setter. Sets the URL where the resource is located at. Used for
            manually downloading corpora.
        """
        self._url = url if url else ''

    @property
    def corpus(self):
        """
            Corpus getter.
        """
        return self._corpus

    @corpus.setter
    def corpus(self, corpus):
        """
            Corpus setter. Sets the internal class containing the sentences.
        """
        self._corpus = corpus

    def _download_corpus(self):
        """
            Download the corpus if it has not been downloaded yet.
        """
        try:
            self._download_from_nltk()
        except AttributeError:
            self._download_from_url(f'{self.__class__.__name__}.txt')

    def _download_from_nltk(self):
        """
            Check whether or not the current corpus is already on NLTK
            corpora folder. Download the corpus if it is not available.
        """
        name = self.corpus.__name__
        try:
            nltk.data.find(f'corpora/{name}')
            print(f'natlang.postagging.{self.__class__.__name__}: found corpus'
                  f' locally, there is no need to download it')
        except LookupError:
            nltk.download(name, quiet=True)
            print(f'natlang.postagging.{self.__class__.__name__}: downloading '
                  f'from NLTK')

    def _download_from_url(self, filename):
        """
            Try to create the folder to save the corpus. If the folder
            already exists, there is no need to download the corpus.
            Otherwise, call wget on the url.
        """
        try:
            os.stat(f'{self.folder}/{filename}')
            print(f'natlang.postagging.{self.__class__.__name__}: found corpus'
                  f' locally, there is no need to download it')
        except FileNotFoundError:
            print(f'natlang.postagging.{self.__class__.__name__}: downloading '
                  f'from URL')
            os.makedirs(self.folder, exist_ok=True)
            wget.download(self.url, out=f'{self.folder}/{filename}')

    def _read_corpus(self):
        """
            Read the corpus if it has not yet been read.
        """
        pass

    def _to_universal(self, filename):
        """
            Convert the tags in the corpus to the universal tagset.
        """

        self.mapping = self._universal_mapping()
        try:
            # Circumventing NLTK's lack exception when file does not exist
            os.stat(f'{self.folder}/{filename}')
            print(f'natlang.postagging.{self.__class__.__name__}: reading '
                  f'corpus previously mapped to universal tagset')
            self.corpus = nltk.corpus.TaggedCorpusReader(
                root=self.folder,
                fileids=filename,
                sep='_',
                word_tokenizer=nltk.WhitespaceTokenizer(),
                encoding='utf-8')
            self.tagged_sents = self.corpus.tagged_sents
        except FileNotFoundError:
            print(f'natlang.postagging.{self.__class__.__name__}: mapping '
                  f'the corpus since the mapped version is not available '
                  f'in folder {self.folder}')
            self.mapped_tagged_sents = self.map_corpus_tags()
            self.tagged_sents = self.mapped_tagged_sents
            self.write(filename)

    @staticmethod
    @abc.abstractmethod
    def _universal_mapping():
        """
            Method to provide the mapping between the original tagset
            and the universal tagset that should be provided by the
            user of a corpus.
        """

    def write(self, filename):
        """
            Write the corpus to a txt file. Useful for saving the same
            corpus using different mappings and avoid mapping upon each
            instantiation.
        """
        with open(f'{self.folder}/{filename}', 'w', encoding='utf-8') as file:
            for sent in self.tagged_sents:
                for word, tag in sent:
                    file.write(f'{word}_{tag} ')
                file.write('\n')

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
            return word, self.default_tag

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
            mapping dictionary. Uses a LazyMap from nltk.collections
            to avoid storing the entire corpus in memory.

            :return: entire corpus mapped to the tagset present in
                mapping
        """
        return map(self.map_sentence_tags, self.corpus.tagged_sents())
