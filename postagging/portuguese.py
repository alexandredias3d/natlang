"""
    Module that provides several classes to handle portuguese corpora.
"""
import nltk

from corpus import Corpus


class MacMorpho(Corpus):
    """
        Class to manage the MacMorpho corpus.
    """

    def __init__(self, folder='corpus', universal=True):
        super().__init__(folder=folder, corpus=nltk.corpus.mac_morpho,
                         universal=universal)

    @staticmethod
    def _universal_mapping():
        """
            Provide a mapping from Mac-Morpho tagset to the Universal
            Tagset. The Mac-Morpho tagset was obtained using a set to
            store only unique occurrences of tags directly from the
            corpus.

            Mac-Morpho manual:
            http://nilc.icmc.usp.br/macmorpho/macmorpho-manual.pdf

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
        Class to manage the Floresta corpus.
    """

    def __init__(self, folder='corpus', universal=True):
        super().__init__(folder=folder, corpus=nltk.corpus.floresta,
                         universal=universal)

    @staticmethod
    def _universal_mapping():
        """
            Provide a mapping from Floresta tagset to the Universal
            Tagset. The Floresta tagset was obtained using a set to
            store only unique occurrences of tags directly from the
            corpus.

            Floresta manual:
            https://www.linguateca.pt/Floresta/
            http://visl.sdu.dk/visl/pt/symbolset-floresta.html

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
            Drop syntatic information to keep only the POS tag.

            :param tag str: Floresta tag that contains syntatic
                information followed by the PoS tag
            :return: PoS tag
        """
        return tag.split('+')[-1]


class LacioWeb(Corpus):
    """
        Class to manage the LacioWeb dataset.
    """

    def __init__(self, name=None, folder='corpus', universal=True):
        """
            LacioWeb class constructor.

            :param name str: name of the corpus to be used
            :param universal bool: True if the tagset must be mapped to the
                universal tagset, False otherwise
        """
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
            Corpus name getter.
        """
        return self._name

    @name.setter
    def name(self, name):
        """
            Corpus name setter. LacioWeb provide three corpora:
            journalistic, literary, and didactic. The default
            option will download all of them.
        """
        self._name = name if name else 'full'

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
            self.name = valid[self.name]
        except KeyError:
            raise Exception(f'natlang.postagging.{self.__class__.__name__}: '
                            f'invalid corpus name. Valid options are: full, '
                            f'journalistic, literary, and didactic.')

    def _download_from_url(self, filename):
        """
            Format URL and call super method for download.
        """
        self.url = self.url.format(self.name)
        super()._download_from_url(
            f'{self.__class__.__name__}_{self.name}.txt')

    def _read_corpus(self):
        """
            Read the corpus using NLTK's TaggedCorpusReader.

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
