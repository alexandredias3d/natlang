"""
    Module that manages the Floresta corpus from Linguateca. The corpus
    is available in NLTK. The module checks if the corpus is available
    locally and download it if needed. Provide a function to convert
    from the original tagset to the universal tagset.
"""
import nltk

from corpus import Corpus


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
