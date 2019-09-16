"""
    Provides easier ways to deal with single corpus from NLTK. It is
    possible to subclass the abstract Corpus class to download
    unavailable corpus in NLTK, similar to the LacioWeb class.
"""
import abc
import os

import nltk
import wget


class Corpus(abc.ABC):
    """
        Provides functions for managing corpus such as mapping between
        different tagsets, automatically download, and so on.

        :param corpus TaggedCorpusReader: corpus reader from NLTK
        :param default str: default tag to be used if key is not found
        :param mapping dict: dictionary to map from one tagset to another
        :param folder str: corpus folder name
    """

    @abc.abstractmethod
    def __init__(self, corpus=None, default_tag=None, folder=None, url=None,
                 universal=None):
        self.default_tag = default_tag
        self.folder = folder
        self.url = url
        self.corpus = corpus
        self.mapping = None
        self.mapped_tagged_sents = []

        os.makedirs(self.folder, exist_ok=True)
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
            Downloads the corpus if it has not been downloaded yet.
        """
        try:
            self._download_from_nltk()
        except (AttributeError, ValueError):
            print(f'natlutil.postagging.{self.__class__.__name__}: did not '
                  f'find the given corpus on NLTK')
            self._download_from_url(f'{self.__class__.__name__}.txt')

    def _download_from_nltk(self):
        """
            Downloads the corpus from NLTK if it has not been found
            locally.
        """
        try:
            name = self.corpus.root.path.split('/')[-1]
            nltk.data.find(f'corpora/{name}')
            print(f'natlutil.postagging.{self.__class__.__name__}: found corpus'
                  f' locally, there is no need to download it')
        except LookupError:
            print(f'natlutil.postagging.{self.__class__.__name__}: downloading '
                  f'from NLTK')
            name = self.corpus.__name__
            nltk.download(name, quiet=True, raise_on_error=True)

    def _download_from_url(self, filename):
        """
            Downloads the specified corpus combining the URL and
            filename if it has not been found locally.

            :param filename str: corpus filename
        """
        try:
            os.stat(f'{self.folder}/{filename}')
            print(f'natlutil.postagging.{self.__class__.__name__}: found corpus'
                  f' locally, there is no need to download it')
        except FileNotFoundError:
            print(f'natlutil.postagging.{self.__class__.__name__}: downloading '
                  f'from URL')
            wget.download(self.url, out=f'{self.folder}/{filename}')

    def _read_corpus(self):
        """
            Reads the corpus if it has not yet been read.
        """
        pass

    def _to_universal(self, filename):
        """
            Converts the tags in the corpus to the universal tagset.
        """

        self.mapping = self._universal_mapping()
        try:
            # Circumventing NLTK's lack exception when file does not exist
            os.stat(f'{self.folder}/{filename}')
            print(f'natlutil.postagging.{self.__class__.__name__}: reading '
                  f'corpus previously mapped to universal tagset')
            self.corpus = nltk.corpus.TaggedCorpusReader(
                root=self.folder,
                fileids=filename,
                sep='_',
                word_tokenizer=nltk.WhitespaceTokenizer(),
                encoding='utf-8')
            self.tagged_sents = self.corpus.tagged_sents
        except FileNotFoundError:
            print(f'natlutil.postagging.{self.__class__.__name__}: mapping '
                  f'the corpus since the mapped version is not available '
                  f'in folder {self.folder}')
            self.mapped_tagged_sents = self.map_corpus_tags()
            self.tagged_sents = self.mapped_tagged_sents
            self.write(filename)

    @staticmethod
    @abc.abstractmethod
    def _universal_mapping():
        """
            Provides mapping to the Universal Tagset.
        """

    def write(self, filename):
        """
            Writes the corpus to a txt file. Useful for saving the same
            corpus using different mappings and avoid mapping upon each
            instantiation.

            :param filename str: output file
        """
        with open(f'{self.folder}/{filename}', 'w', encoding='utf-8') as file:
            for sent in self.tagged_sents:
                for word, tag in sent:
                    if word and tag:
                        file.write(f'{word}_{tag} ')
                file.write('\n')

    def map_word_tag(self, word_tag):
        """
            Maps a single word-tag tuple to the tagset present in
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
            Maps tags from a sentence to the tagset present in the
            mapping dictionary.

            :param sentence list: list of word-tag tuples
            :return: list of word-tag tuples, where the tags have
                been mapped to the tagset in mapping
        """
        return list(map(self.map_word_tag, sentence))

    def map_corpus_tags(self):
        """
            Maps the entire corpus to the tagset present in the
            mapping dictionary.

            :return: entire corpus mapped to the tagset present in
                mapping
        """
        return map(self.map_sentence_tags, self.corpus.tagged_sents())
