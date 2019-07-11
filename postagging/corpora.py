import itertools

from portuguese import Floresta, LacioWeb, MacMorpho


class Corpora:
    """
        Class to easily manage corpora.
    """

    def __init__(self, corpora, universal=True):
        """
            Class constructor. Receive a list of corpora names and a
            boolean to control universal mapping.

            :param corpora list: list of strings representing corpus' names
            :param universal bool: True if original tagset should be converted
                to the universal tagset, False otherwise
        """
        self.sentences = None
        self._length = 0
        self._validate_corpus_names(corpora)
        for corpus in corpora:
            self._read_corpus(corpus, universal)

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

    @staticmethod
    def _validate_corpus_names(corpora):
        """
            Map a string name into the available corpus
            class in natlang.

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
            raise Exception('natlang.postagging.corpus: invalid corpus name')

    def _read_corpus(self, cls, universal=True):
        """
            Read the given corpus and convert it to universal tagset
            if universal is set to True.

            :param cls:
            :param universal bool:
        """
        sents = cls(universal=universal).corpus.tagged_sents()
        self._length = self._length + len(sents)
        self.sentences = self.sentences + sents if self.sentences else sents

    @staticmethod
    def _get_words(sentence):
        """
            Get the words from the given tagged sentence.
        """
        return [word for word, tag in sentence]

    @staticmethod
    def _get_tags(sentence):
        """
            Get the tags from the given tagged sentence.
        """
        return [tag for word, tag in sentence]

    def train_test_split(self, test_size=0.3):
        """
            Split the corpora into training and test data.
        """
        train_size = 1 - test_size
        total_size = len(self.sentences)

        x_train = map(self._get_words,
                      itertools.islice(self.sentences,
                                       0,
                                       total_size // int(train_size * 100)))

        x_test = map(self._get_words,
                     itertools.islice(self.sentences,
                                      total_size // int(train_size * 100),
                                      total_size))

        y_train = map(self._get_tags,
                      itertools.islice(self.sentences,
                                       0,
                                       total_size // int(train_size * 100)))
        y_test = map(self._get_tags,
                     itertools.islice(self.sentences,
                                      total_size // int(train_size * 100),
                                      total_size))

        return x_train, x_test, y_train, y_test
