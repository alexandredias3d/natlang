from itertools import chain, islice
from natlutil.pos.portuguese import Floresta, LacioWeb, MacMorpho


class Corpora:
    """
        Class to easily manage corpora.
    """

    def __init__(self, corpora, folder='corpus', universal=True):
        """
            Class constructor. Receive a list of corpora names and a
            boolean to control universal mapping.

            :param corpora list: list of strings representing corpus' names
            :param folder str: the folder to load and store corpora
            :param universal bool: True if original tagset should be converted
                to the universal tagset, False otherwise
        """

        self.sentences = None
        self._length = 0
        self._validate_corpus_names(corpora)
        for corpus in corpora:
            self._read_corpus(corpus, folder, universal)

    def __len__(self):
        """
            Shortcut to the amount of sentences in the corpora.
        """
        return self._length

    def __getitem__(self, key):
        """
            Shortcut for indexing the sentences.
        """
        return self.sentences[key]

    def __iter__(self):
        """
            Shortcut for getting an iterator over the sentences.
        """
        return chain(self.sentences)

    @staticmethod
    def _validate_corpus_names(corpora):
        """
            Maps a string name into the available corpus
            class in natlutil.

            :param corpora list:
        """
        mapping = {'MACMORPHO': MacMorpho,
                   'FLORESTA': Floresta,
                   'LACIOWEB': LacioWeb}

        def _clean(name):
            return name.upper().replace(' ', '').replace('_', '')

        try:
            for idx, name in enumerate(corpora):
                corpora[idx] = mapping[_clean(name)]
        except KeyError:
            raise Exception('natlutil.postagging.corpus: invalid corpus name')

    def _read_corpus(self, cls, folder=None, universal=True):
        """
            Reads the given corpus and converts it to universal tagset
            if universal is set to True.

            :param cls:
            :param universal bool:
        """
        sents = cls(folder=folder, universal=universal).corpus.tagged_sents()
        self._length = self._length + len(sents)
        self.sentences = self.sentences + sents if self.sentences else sents

    def get_train(self, train_size=0.7):
        """
            Gets the training set.

            :param train_size double: amount of sentences (in percentage)
                in the training set
            :return: iterable over the sentences
        """
        return islice(self.sentences, 0, int(len(self) * train_size))

    def get_test(self, test_size=0.3):
        """
            Gets the test set.

            :param test_size double: amount of sentences (in percentage)
                in the test set
            :return: iterable over the sentences
        """
        return islice(self.sentences, int(len(self) - (len(self) * test_size)),
                      len(self))
