"""
    Module that manages the MacMorpho corpus from NILC. The corpus is
    available in NLTK. The module checks if the corpus is available
    locally and download it if needed. Provide a function to convert
    from the original tagset to the universal tagset.
"""
import nltk

from corpus import Corpus


class MacMorpho(Corpus):
    """
        Class to manage the MacMorpho corpus.
    """

    def __init__(self, universal=True):
        super().__init__(corpus=nltk.corpus.mac_morpho)
        if universal:
            self._to_universal()

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
