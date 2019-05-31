import nltk

class Floresta:
    '''
        Class to manage the Floresta corpus.
    '''

    corpus = nltk.corpus.floresta
    name = 'floresta'

    def __init__(self, universal=True):
        if universal:
            self.unitags = self._universal_mapping()
            self.universal_tagged_sents = self.map_corpus_tags()

    @classmethod
    def _universal_mapping(cls):
        '''
            Provide a mapping from Floresta tagset to the Universal
            Tagset. The Floresta tagset was obtained using a set to
            store only unique occurrences of tags.

            Floresta manual:
            https://www.linguateca.pt/Floresta/
            http://visl.sdu.dk/visl/pt/symbolset-floresta.html
        '''
        unitags = {}

        # Punctuation: .
        unitags.update({k: '.' for k in ['!', '"', "'", '*', ',', '-', '.',
                                         '/', ';', '?', '[', ']', '{', '}',
                                         '»', '«']})

        # Adjectives: ADJ
        unitags.update({k: 'ADJ' for k in ['adj']})

        # Numbers: NUM
        unitags.update({k: 'NUM' for k in ['num']})

        # Adverbs: ADV
        unitags.update({k: 'ADV' for k in ['adv']})

        # Conjunctions: CONJ
        unitags.update({k: 'CONJ' for k in ['conj-c', 'conj-s']})

        # Determiners: DET
        unitags.update({k: 'DET' for k in ['art']})

        # Nouns: NOUN
        # prop: proper noun.
        unitags.update({k: 'NOUN' for k in ['n', 'prop']})

        # Pronouns: PRON
        unitags.update({k: 'PRON' for k in ['pron-det', 'pron-indp',
                                            'pron-pers']})

        # Particles: PRT
        # Not present in the tagset.
        #unitags.update({k: 'PRT' for k in ['']})

        # Adpositions: ADP
        # Three occurrences of "em" with tags H+prp-.
        # Should pp (prepositional phrase) be classified as an adposition?
        # Original paper of Universal Tagset classified pp as noun...
        unitags.update({k: 'ADP' for k in ['prp', 'prp-', 'pp']})

        # Verbs: VERB
        # Should participle verbs go here or in adjectives?
        # "existente" has tag P+vp: predicator + verb phrase (should it be
        # tagged as a verb?)
        unitags.update({k: 'VERB' for k in ['v-fin', 'v-ger', 'v-inf', 'v-pcp',
                                            'vp']})

        # Miscellaneous: X
        # N<{'185/60_R_14'} is the tag for the word 185/60_R_14.
        # ec: anti-, ex-, pós, ex, pré (how should they be classified?)
        unitags.update({k: 'X' for k in ['ec', 'in', "N<{'185/60_R_14'}"]})

        return unitags

    @classmethod
    def _get_pos_tag(cls, tag):
        '''
            Drop syntatic information to keep only the POS tag.
        '''
        return tag.split('+')[-1]

    def map_word_tag(self, word, tag):
        '''
            Map a single word-tag tuple to universal tagset.
        '''
        return (word, self.unitags.get(self._get_pos_tag(tag), 'X'))

    def map_sentence_tags(self, sentence):
        '''
            Map tags from a sentence to universal tagset.
        '''
        return [self.map_word_tag(word, tag) for word, tag in sentence]

    def map_corpus_tags(self):
        '''
            Map the entire corpus to universal tagset.
        '''
        return [self.map_sentence_tags(sentence)
                for sentence in self.corpus.tagged_sents()]
