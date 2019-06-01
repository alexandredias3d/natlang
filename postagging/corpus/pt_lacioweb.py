"""
    Module that handles the LacioWeb corpus from NILC. It can download
    and read the corpus using the NLTK TaggedCorpusReader. Also, it can
    convert the original PoS tags to the universal tagset.
"""
import os

import nltk
import wget

from util import Corpus


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
        super().__init__(folder='corpus/lacioweb/')
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
        os.makedirs(self._folder, exist_ok=True)
        wget.download(self._url.format(self._name), out=self._folder)

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
        """
        unitags = {}

        # Punctuation: .
        unitags.update({k: '.' for k in ['!', '"', "'", '(', ')', ',', '-',
                                         '.', '...', ':', ';', '?', '[', ']']})
        # Adjectives: ADJ
        unitags.update({k: 'ADJ' for k in ['ADJ']})

        # Numbers: NUM
        unitags.update({k: 'NUM' for k in ['NC', 'ORD', 'NO']})

        # Adverbs: ADV
        unitags.update({k: 'ADV' for k in ['ADV', 'ADV+PPOA', 'ADV+PPR',
                                           'LADV']})

        # Conjunctions: CONJ
        unitags.update({k: 'CONJ' for k in ['CONJCOORD', 'CONJSUB', 'LCONJ']})

        # Determiners: DET
        unitags.update({k: 'DET' for k in ['ART']})

        # Nouns: NOUN
        unitags.update({k: 'NOUN' for k in ['N', 'NP']})

        # Pronouns: PRON
        unitags.update({k: 'PRON' for k in ['PAPASS', 'PD', 'PIND', 'PINT',
                                            'PPOA', 'PPOA+PPOA', 'PPOT', 'PPR',
                                            'PPS', 'PR', 'PREAL', 'PTRA',
                                            'LP']})

        # Particles: PRT
        unitags.update({k: 'PRT' for k in ['PDEN', 'LDEN']})

        # Adposition: ADP
        unitags.update({k: 'ADP' for k in ['PREP', 'PREP+ADJ', 'PREP+ADV',
                                           'PREP+ART', 'PREP+N', 'PREP+PD',
                                           'PREP+PPOA', 'PREP+PPOT',
                                           'PREP+PPR', 'PREP+PREP', 'LPREP',
                                           'LPREP+ART']})

        '''
            AUX is a typo from VAUX (four occurrences):
                - sendo;
                - continuar;
                - deve;
                - foram_V

            INT is a typo from VINT (only one occurrence):
                - ocorrido.
        '''
        # Verbs: VERB
        unitags.update({k: 'VERB' for k in ['VAUX', 'VAUX!PPOA', 'VAUX+PPOA',
                                            'VBI', 'VBI+PAPASS', 'VBI+PPOA',
                                            'VBI+PPR', 'VINT', 'VINT+PAPASS',
                                            'VINT+PPOA', 'VINT+PREAL', 'VLIG',
                                            'VLIG+PPOA', 'VTD', 'VTD!PPOA',
                                            'VTD+PAPASS', 'VTD+PPOA',
                                            'VTD+PPR', 'VTD+PREAL', 'VTI',
                                            'VTI+PPOA', 'VTI+PREAL', 'AUX',
                                            'INT']})

        '''
            IL should probably be residual there are two occurrences:
                - CL- (in sentence "cloro (CL-):");
                - po4- (in sentence "Fosfato (po4-):").
            It seems to be from a didactic chemistry text.
        '''
        # Miscellaneous: X
        unitags.update({k: 'X' for k in ['I', 'RES', 'IL']})
        return unitags
