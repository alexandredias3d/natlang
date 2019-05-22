import nltk

class MacMorpho:
    '''
        Class to manage the MacMorpho corpus.
    '''

    def __init__(self, universal_tagset=True):
        self.corpus = nltk.corpus.mac_morpho
        if universal_tagset:
            self.unitags = self._universal_mapping()
            self.universal_tagged_sents = self.map_corpus_tags()

    @classmethod
    def _universal_mapping(cls):
        '''
            Provide a mapping from Mac-Morpho tagset to the Universal
            Tagset. The Mac-Morpho tagset was obtained using a set to
            store only unique occurrences of tags.

            Mac-Morpho manual:
            http://nilc.icmc.usp.br/macmorpho/macmorpho-manual.pdf
        '''
        unitags = {}

        # Punctuation: .
        # According to Wikipedia, $ is a punctuation mark.
        unitags.update({k: '.' for k in ['!', '"', '$', "'", '(', ')', ',', '-',
                                         '.', '/', ':', ';', '?', '[', ']']})

        # Adjectives: ADJ
        unitags.update({k: 'ADJ' for k in ['ADJ', 'ADJ|EST']})

        # Numbers: NUM
        unitags.update({k: 'NUM' for k in ['NUM', 'NUM|TEL']})

        # Adverbs: ADV
        unitags.update({k: 'ADV' for k in ['ADV', 'ADV-KS', 'ADV-KS-REL',
                                           'ADV|+', 'ADV|EST', 'ADV|[',
                                           'ADV|]']})
        # Conjunctions: CONJ
        unitags.update({k: 'CONJ' for k in ['KC', 'KC|[', 'KC|]', 'KS']})

        # Determiners: DET
        unitags.update({k: 'DET' for k in ['ART', 'ART|+']})

        # Nouns: NOUN
        # NPRO is a typo: two occurrences for "Folha" and
        #                     one for "Congresso".
        unitags.update({k: 'NOUN' for k in ['N', 'NPRO', 'NPROP', 'NPROP|+',
                                            'N|AP', 'N|DAT', 'N|EST', 'N|HOR',
                                            'N|TEL']})

        # Pronouns: PRON
        unitags.update({k: 'PRON' for k in ['PRO-KS', 'PRO-KS-REL', 'PROADJ',
                                            'PROPESS', 'PROSUB']})

        # Particles: PRT
        unitags.update({k: 'PRT' for k in ['PDEN']})

        # Adpositions: ADP
        # PREP| is a typo of PREP: two occurrences for "de".
        unitags.update({k: 'ADP' for k in ['PREP', 'PREP|', 'PREP|+',
                                           'PREP|[', 'PREP|]']})

        # Verbs: VERB
        # Should participle verbs go here or in adjectives?
        unitags.update({k: 'VERB' for k in ['V', 'V|+', 'VAUX', 'VAUX|+',
                                            'PCP']})

        # Miscellaneous: X
        unitags.update({k: 'X' for k in ['CUR', 'IN']})

        return unitags

    def map_word_tag(self, word_tag):
        '''
            Map a single word-tag tuple to universal tagset.
        '''
        return (word_tag[0], self.unitags.get(word_tag[1], 'X'))

    def map_sentence_tags(self, sentence):
        '''
            Map tags from a sentence to universal tagset.
        '''
        return [(word, self.unitags.get(tag, 'X')) for
                word, tag in sentence]

    def map_corpus_tags(self):
        '''
            Map the entire corpus to universal tagset.
        '''
        return [self.map_sentence_tags(sentence)
                for sentence in self.corpus.tagged_sents()]
