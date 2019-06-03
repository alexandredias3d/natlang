"""
    Module that handles the LacioWeb corpus from NILC. It can download
    and read the corpus using the NLTK TaggedCorpusReader. Also, it can
    convert the original PoS tags to the universal tagset.
"""
import nltk

from corpus import Corpus


class LacioWeb(Corpus):
    """
        Class to manage the LacioWeb dataset.
    """

    def __init__(self, name='full', universal=True):
        """
            LacioWeb class constructor.

            :param name str: name of the corpus to be used
            :param universal bool: True if the tagset must be mapped to the
                universal tagset, False otherwise
        """
        super().__init__(folder='lacioweb/')
        self._url = 'http://nilc.icmc.usp.br/nilc/download/corpus{}.txt'
        self._name = name
        self._validate_corpus_name()
        self._get_corpus()

        if universal:
            self.mapping = self._universal_mapping()
            self.mapped_tagged_sents = self.map_corpus_tags()

    def _validate_corpus_name(self):
        """
            Check if the given name is a valid corpus name.
            Available options are full, journalistic, literary,
            and didactic.
        """
        valid = {'full':         '100',
                 'journalistic': 'journalistic',
                 'literary':     'literary',
                 'didactic':     'didactic'}

        try:
            self._name = valid[self._name]
        except KeyError:
            raise Exception('natlang.postagging.pt_lacioweb: invalid corpus '
                            'name. Valid options are: full, journalistic, '
                            'literary, and didactic.')

    def _get_corpus(self):
        """
            Read the corpus. If the an OSError is raised, it means that there
            is the need to download the corpus.
        """
        filename = f'corpus{self._name}.txt'
        try:
            self._read_corpus(filename)
        except OSError:
            self._download_from_url()
            self._read_corpus(filename)

    def _download_from_url(self):
        """
            Download the corpus in the given url. Create folder if it
            does not exist.
        """
        self._url = self._url.format(self._name)
        super()._download_from_url()

    def _read_corpus(self, filename):
        """
            Read the corpus using NLTK's TaggedCorpusReader.

            :param filename str: corpus filename
        """
        self.corpus = nltk.corpus.TaggedCorpusReader(
            root=self._folder,
            fileids=filename,
            sep='_',
            word_tokenizer=nltk.WhitespaceTokenizer(),
            encoding='latin-1')

    @staticmethod
    def _universal_mapping():
        """
            Provide a mapping from the NILC tagset used in the
            LacioWeb corpus. The following tags were directly extracted
            from the data and inconsistencies were analyzed.

            NILC tagset:
            http://www.nilc.icmc.usp.br/nilc/download/tagsetcompleto.doc
            
            :return: dictionary that maps the current tagset to the
                universal tagset
        """
        mapping = {

            # Punctuation: .
            **{k: '.' for k in ['!', '"', "'", '(', ')', ',', '-', '.', '...',
                                ':', ';', '?', '[', ']']},

            # Adjectives: ADJ
            **{k: 'ADJ' for k in ['ADJ']},

            # Numbers: NUM
            **{k: 'NUM' for k in ['NC', 'ORD', 'NO']},

            # Adverbs: ADV
            **{k: 'ADV' for k in ['ADV', 'ADV+PPOA', 'ADV+PPR', 'LADV']},

            # Conjunctions: CONJ
            **{k: 'CONJ' for k in ['CONJCOORD', 'CONJSUB', 'LCONJ']},

            # Determiners: DET
            **{k: 'DET' for k in ['ART']},

            # Nouns: NOUN
            **{k: 'NOUN' for k in ['N', 'NP']},

            # Pronouns: PRON
            **{k: 'PRON' for k in ['PAPASS', 'PD', 'PIND', 'PINT', 'PPOA',
                                   'PPOA+PPOA', 'PPOT', 'PPR', 'PPS', 'PR',
                                   'PREAL', 'PTRA', 'LP']},

            # Particles: PRT
            **{k: 'PRT' for k in ['PDEN', 'LDEN']},

            # Adposition: ADP
            **{k: 'ADP' for k in ['PREP', 'PREP+ADJ', 'PREP+ADV', 'PREP+ART',
                                  'PREP+N', 'PREP+PD', 'PREP+PPOA',
                                  'PREP+PPOT', 'PREP+PPR', 'PREP+PREP',
                                  'LPREP', 'LPREP+ART']},

            # Verbs: VERB
            # AUX is a typo from VAUX (four occurrences):
            #   - sendo, continuar, deve, foram_V.
            #
            # INT is a typo from VINT (only one occurrence):
            #   - ocorrido.
            **{k: 'VERB' for k in ['VAUX', 'VAUX!PPOA', 'VAUX+PPOA', 'VBI',
                                   'VBI+PAPASS', 'VBI+PPOA', 'VBI+PPR',
                                   'VINT', 'VINT+PAPASS', 'VINT+PPOA',
                                   'VINT+PREAL', 'VLIG', 'VLIG+PPOA', 'VTD',
                                   'VTD!PPOA', 'VTD+PAPASS', 'VTD+PPOA',
                                   'VTD+PPR', 'VTD+PREAL', 'VTI', 'VTI+PPOA',
                                   'VTI+PREAL', 'AUX', 'INT']},

            # Miscellaneous: X
            # IL should probably be residual. There are two occurrences:
            #   - CL- (in sentence "cloro (CL-):");
            #   - po4- (in sentence "Fosfato (po4-):").
            **{k: 'X' for k in ['I', 'RES', 'IL']}}

        return mapping
