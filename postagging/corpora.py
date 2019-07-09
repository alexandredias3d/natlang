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
        self.tagged_sents = None
        self._validate_corpus_names(corpora)
        for corpus in corpora:
            self._read_corpus(corpus, universal)

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
        corpus = cls(universal=universal).corpus
        self.tagged_sents = self.tagged_sents + corpus.tagged_sents() \
            if self.tagged_sents else corpus.tagged_sents()
