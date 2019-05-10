import nltk
import os
import requests

class LacioWeb:

    def __init__(self):
        self.root_path = 'corpus/lacioweb'
        self.root_url = 'http://nilc.icmc.usp.br/nilc/download/corpus{}.txt'
        self.corpus = {}

        self._universal_mapping()

    def _download_corpus(self, name):
        '''
            Download the corpus in the given url.
        '''
        r = requests.get(self.root_url.format(name))
        with open(f'{self.root_path}/corpus{name}.txt', 'w') as file:
            file.write(r.text)

    def _validate_corpus_name(self, name):
        '''
            Check if the given name is a valid corpus name.
            Available options are full, journalistic, literary,
            and didactic.
        '''
        d = {'full': '100', 
             'journalistic': 'journalistic',
             'literary': 'literary',
             'didactic': 'didactic'}

        error_msg = '''
                    Valid names are: full, journalistic, 
                    literary, and didactic.
                    '''

        if name in d:
            return d[name]
        else:
            raise Exception(error_msg)

    def get_corpus(self, names):
        '''
            Get each corpus correctly named in names. Available options are
            full, journalistic, literary, and didactic.
        '''
        os.makedirs(self.root_path, exist_ok=True)
        
        if isinstance(names, list):
            for name in names:
                self._download_corpus(self._validate_corpus_name(name))
        else:
            self._download_corpus(self._validate_corpus_name(names))
                
    def read_corpus(self, name):
        '''
            Read the corpus using NLTK's TaggedCorpusReader.
        '''
        filename = f'corpus{self._validate_corpus_name(name)}.txt'
        if os.path.exists(f'{self.root_path}/{filename}'):
            self.corpus[name] = nltk.corpus.TaggedCorpusReader(
                                    root=self.root_path,
                                    fileids=filename,
                                    sep='_',
                                    word_tokenizer=nltk.WhitespaceTokenizer())
        else:
            raise FileNotFoundError('Could not find the corpus.')

    def _universal_mapping(self):
        '''
            Provide a mapping from the NILC tagset used in the
            LacioWeb corpus. The following tags were directly extracted
            from the data and inconsistencies were analyzed. 
            
            NILC tagset:
            http://www.nilc.icmc.usp.br/nilc/download/tagsetcompleto.doc
        '''
        d = {}

        # Punctuation: .
        d.update({k: '.' for k in ['!', '"', "'", '(', ')', ',', '-', '.',
                                       '...', ':', ';', '?', '[', ']']})
        # Adjectives: ADJ
        d.update({k: 'ADJ' for k in ['ADJ']})

        # Numbers: NUM
        d.update({k: 'NUM' for k in ['NC', 'ORD', 'NO']})

        # Adverbs: ADV
        d.update({k: 'ADV' for k in ['ADV', 'ADV+PPOA', 'ADV+PPR', 'LADV']})

        # Conjunctions: CONJ
        d.update({k: 'CONJ' for k in ['CONJCOORD', 'CONJSUB', 'LCONJ']})

        # Determiners: DET
        d.update({k: 'DET' for k in ['ART']})

        # Nouns: NOUN
        d.update({k: 'NOUN' for k in ['N', 'NP']})

        # Pronouns: PRON
        d.update({k: 'PRON' for k in ['PAPASS', 'PD', 'PIND', 'PINT',
                     'PPOA', 'PPOA+PPOA', 'PPOT', 'PPR', 'PPS',
                     'PR', 'PREAL', 'PTRA', 'LP']})

        # Particles: PRT
        d.update({k: 'PRT' for k in ['PDEN', 'LDEN']})

        # Adposition: ADP
        d.update({k: 'ADP' for k in ['PREP', 'PREP+ADJ', 'PREP+ADV',
                     'PREP+ART', 'PREP+N', 'PREP+PD', 'PREP+PPOA',
                     'PREP+PPOT', 'PREP+PPR', 'PREP+PREP', 'LPREP',
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
        d.update({k: 'VERB' for k in ['VAUX', 'VAUX!PPOA', 'VAUX+PPOA',
                     'VBI', 'VBI+PAPASS', 'VBI+PPOA', 'VBI+PPR', 'VINT',
                     'VINT+PAPASS', 'VINT+PPOA', 'VINT+PREAL', 'VLIG',
                     'VLIG+PPOA', 'VTD', 'VTD!PPOA', 'VTD+PAPASS', 
                     'VTD+PPOA', 'VTD+PPR', 'VTD+PREAL', 'VTI', 'VTI+PPOA',
                     'VTI+PREAL', 'AUX', 'INT']})

        '''
            IL should probably be residual there are two occurrences:
                - CL- (in sentence "cloro (CL-):");
                - po4- (in sentence "Fosfato (po4-):").
            It seems to be from a didactic chemistry text.
        '''
        # Miscellaneous: X
        d.update({k: 'X' for k in ['I', 'RES', 'IL']})
        self.mapping = d

    def map_word_tag(self, word_tag):
        '''
            Map a single word-tag tuple to universal tagset.
        '''
        return (word_tag[0], self.mapping.get(word_tag[1], 'X'))

    def map_sentence_tags(self, sentence):
        '''
            Map tags from a sentence to universal tagset.
        '''
        return [(word_tag[0], self.mapping.get(word_tag[1], 'X')) for
                word_tag in sentence]

    def map_corpus_tags(self):
        '''
            Map entire corpus to universal tagset.
        '''
        self.universal_tagged_sents = {}
        for corpus in self.corpus:
            self.universal_tagged_sents[corpus] = [
                self.map_sentence_tags(sentence) for sentence in 
                self.corpus[corpus].tagged_sents()]
