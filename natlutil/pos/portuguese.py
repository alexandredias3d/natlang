"""
    Provides several classes to handle portuguese corpora.
"""
import nltk

from natlutil.pos.corpus import Corpus


class MacMorpho(Corpus):
    """
        Manages the MacMorpho corpus.

        Mac-Morpho tagset manual:
        http://nilc.icmc.usp.br/macmorpho/macmorpho-manual.pdf

        :param folder str: output folder
        :param universal bool: True if the tagset must be mapped to the
            universal tagset, False otherwise
    """

    def __init__(self, folder='corpus', universal=True):
        super().__init__(folder=folder, corpus=nltk.corpus.mac_morpho,
                         universal=universal)

    @staticmethod
    def _universal_mapping():
        """
            Provides mapping to the Universal Tagset.

            :return: dictionary that maps the current tagset to the
                universal tagset
        """
        mapping = {

            # Punctuation: .
            # According to Wikipedia, $ is a punctuation mark.
            **{k: '.' for k in ['!', '"', '$', "'", '(', ')', ',', '-',
                                '.', '/', ':', ';', '?', '[', ']']},

            # Adjectives: ADJ
            **{k: 'ADJ' for k in ['ADJ', 'ADJ|EST']},

            # Numbers: NUM
            **{k: 'NUM' for k in ['NUM', 'NUM|TEL']},

            # Adverbs: ADV
            **{k: 'ADV' for k in ['ADV', 'ADV-KS', 'ADV-KS-REL', 'ADV|+',
                                  'ADV|EST', 'ADV|[', 'ADV|]']},
            # Conjunctions: CONJ
            **{k: 'CONJ' for k in ['KC', 'KC|[', 'KC|]', 'KS']},

            # Determiners: DET
            **{k: 'DET' for k in ['ART', 'ART|+']},

            # Nouns: NOUN
            # NPRO is a typo: two occurrences for "Folha" and
            #                     one for "Congresso".
            **{k: 'NOUN' for k in ['N', 'NPRO', 'NPROP', 'NPROP|+', 'N|AP',
                                   'N|DAT', 'N|EST', 'N|HOR', 'N|TEL']},

            # Pronouns: PRON
            **{k: 'PRON' for k in ['PRO-KS', 'PRO-KS-REL', 'PROADJ', 'PROPESS',
                                   'PROSUB']},

            # Particles: PRT
            **{k: 'PRT' for k in ['PDEN']},

            # Adpositions: ADP
            # PREP| is a typo of PREP: two occurrences for "de".
            **{k: 'ADP' for k in ['PREP', 'PREP|', 'PREP|+', 'PREP|[',
                                  'PREP|]']},

            # Verbs: VERB
            # Should participle verbs go here or in adjectives?
            **{k: 'VERB' for k in ['V', 'V|+', 'VAUX', 'VAUX|+', 'PCP']},

            # Miscellaneous: X
            **{k: 'X' for k in ['CUR', 'IN']}}

        return mapping


class Floresta(Corpus):
    """
        Manages the Floresta corpus.

        Floresta tagset manual:
        https://www.linguateca.pt/Floresta/
        http://visl.sdu.dk/visl/pt/symbolset-floresta.html

        :param folder str: output folder
        :param universal bool: True if the tagset must be mapped to the
            universal tagset, False otherwise
    """

    def __init__(self, folder='corpus', universal=True):
        super().__init__(folder=folder, corpus=nltk.corpus.floresta,
                         universal=universal)

    @staticmethod
    def _universal_mapping():
        """
            Provides mapping to the Universal Tagset.

            :return: dictionary that maps the current tagset to the
                universal tagset
        """
        mapping = {

            # Punctuation: .
            **{k: '.' for k in ['!', '"', "'", '*', ',', '-', '.', '/',
                                ';', '?', '[', ']', '{', '}', '»', '«']},

            # Adjectives: ADJ
            **{k: 'ADJ' for k in ['adj']},

            # Numbers: NUM
            **{k: 'NUM' for k in ['num']},

            # Adverbs: ADV
            **{k: 'ADV' for k in ['adv']},

            # Conjunctions: CONJ
            **{k: 'CONJ' for k in ['conj-c', 'conj-s']},

            # Determiners: DET
            **{k: 'DET' for k in ['art']},

            # Nouns: NOUN
            # prop: proper noun.
            **{k: 'NOUN' for k in ['n', 'prop']},

            # Pronouns: PRON
            **{k: 'PRON' for k in ['pron-det', 'pron-indp', 'pron-pers']},

            # Particles: PRT
            # Not present in the tagset.
            **{k: 'PRT' for k in ['']},

            # Adpositions: ADP
            # Three occurrences of "em" with tags H+prp-.
            # Should pp (prepositional phrase) be classified as an adposition?
            # Original paper of Universal Tagset classified pp as noun...
            **{k: 'ADP' for k in ['prp', 'prp-', 'pp']},

            # Verbs: VERB
            # Should participle verbs go here or in adjectives?
            # "existente" has tag P+vp: predicator + verb phrase (should it be
            # tagged as a verb?)
            **{k: 'VERB' for k in ['v-fin', 'v-ger', 'v-inf', 'v-pcp', 'vp']},

            # Miscellaneous: X
            # N<{'185/60_R_14'} is the tag for the word 185/60_R_14.
            # ec: anti-, ex-, pós, ex, pré (how should they be classified?)
            **{k: 'X' for k in ['ec', 'in', "N<{'185/60_R_14'}"]}}

        return mapping

    def map_word_tag(self, word_tag):
        return super().map_word_tag((word_tag[0],
                                     self._get_pos_tag(word_tag[1])))

    @staticmethod
    def _get_pos_tag(tag):
        """
            Drops syntatic information to keep only the POS tag.

            :param tag str: Floresta tag that contains syntatic
                information followed by the PoS tag
            :return: PoS tag
        """
        return tag.split('+')[-1]


class LacioWeb(Corpus):
    """
        Manages the LacioWeb dataset.

        NILC tagset manual:
        http://www.nilc.icmc.usp.br/nilc/download/tagsetcompleto.doc

        :param name str: name of the corpus to be used
        :param folder str: output folder
        :param universal bool: True if the tagset must be mapped to the
            universal tagset, False otherwise
    """

    def __init__(self, name=None, folder='corpus', universal=True):
        # Python constructors are different from those in Java: it is possible
        # to call methods before the constructor since it is only a method that
        # returns a proxy object for accessing parent attributes.
        self.name = name
        self._validate_corpus_name()

        super().__init__(
            folder=folder,
            url='http://nilc.icmc.usp.br/nilc/download/corpus{}.txt',
            universal=universal)

    @property
    def name(self):
        """
            Gets the corpus name.
        """
        return self._name

    @name.setter
    def name(self, name):
        """
            Sets the corpus name if the given one is available or use
            full as the default name.
        """
        self._name = name if name else 'full'

    def _validate_corpus_name(self):
        """
            Checks if the given name is a valid corpus name.
            Available options are full, journalistic, literary,
            and didactic.
        """
        valid = {'full':         '100',
                 'journalistic': 'journalistic',
                 'literary':     'literary',
                 'didactic':     'didactic'}

        try:
            self.name = valid[self.name]
        except KeyError:
            raise Exception(f'natlutil.postagging.{self.__class__.__name__}: '
                            f'invalid corpus name. Valid options are: full, '
                            f'journalistic, literary, and didactic.')

    def _download_from_url(self, filename):
        """
            Formats URL and calls super method for download.

            :param filename str: name of the file to be retrieved from
                the URL
        """
        self.url = self.url.format(self.name)
        super()._download_from_url(
            f'{self.__class__.__name__}_{self.name}.txt')

    def _read_corpus(self):
        """
            Reads the corpus using NLTK's TaggedCorpusReader.

            :param filename str: corpus filename
        """
        self.corpus = nltk.corpus.TaggedCorpusReader(
            root=self.folder,
            fileids=f'{self.__class__.__name__}_{self.name}.txt',
            sep='_',
            word_tokenizer=nltk.WhitespaceTokenizer(),
            encoding='latin-1')

    def _to_universal(self, filename):
        super()._to_universal(
            f'{self.__class__.__name__}_{self.name}_Universal.txt')

    @staticmethod
    def _universal_mapping():
        """
            Provides mapping to the Universal Tagset.

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
