import nltk
import pickle

class TaggerFactory:
    '''
        Tagger factory that abstracts the creation of taggers.
        The user should still know how to use NLTK taggers
        since they must provide arguments to the function.
    '''

    @staticmethod
    def sequence_backoff_factory(*args):
        '''
            Given a list of tuples conntaining tagger names and 
            arguments, generate a sequence to be used in the 
            SequenceBackoffTagger. The first name should be the
            first tagger to be called, which will backoff to the 
            second tagger and so on.

        '''
        mapping = {'DEFAULT': nltk.DefaultTagger,
                   'UNIGRAM': nltk.UnigramTagger,
                   'BIGRAM' : nltk.BigramTagger,
                   'TRIGRAM': nltk.TrigramTagger,
                   'NGRAM'  : nltk.NgramTagger,
                   'AFFIX'  : nltk.AffixTagger,
                   'REGEX'  : nltk.RegexpTagger,
                   'REGEXP' : nltk.RegexpTagger}

        def _clean(name):
            return mapping[name.upper().replace('TAGGER', '')]

        return [(_clean(name), arguments) for name, arguments in args]

class SequenceBackoffTagger:
    def __init__(self, *args, reverse=True):
        '''
            Class constructor. Receives a sequence from the 
            sequence_backoff_factory method and creates the
            given tagger combination.
        '''
        self.taggers = []
        
        sequence = reversed(*args) if reverse else args
        for cls, arguments in sequence:
            if issubclass(cls, nltk.DefaultTagger):
                self.taggers.append(cls(arguments))
            else:
                arguments['backoff'] = self.taggers[-1]
                self.taggers.append(cls(**arguments))

        self.tagger = self.taggers[-1]

    def evaluate(self, test_set):
        '''
            Evaluate the combined tagger on the given test set.
        '''
        print(self.tagger.evaluate(test_set))

    def evaluate_all(self, test_set):
        '''
            Evaluate and print the accuracy for each tagger.
            Combined taggers get improved performance due to
            the backoff sequence.
        '''
        headers = ['Tagger', 'Accuracy']
        row = '{:<10}' * len(headers)
        print(row.format(*headers))
        row = '{:<10}{:<10.4f}'
        for tagger in self.taggers:
            print(row.format(tagger.__class__.__name__.replace('Tagger', ''), 
                             tagger.evaluate(test_set)))

    def save(self, filepath):
        '''
            Save a trained tagger for future use. This method will only
            save the final tagger.
        '''
        if not filepath.endswith('.pkl'):
            filepath += '.pkl'
        with open(f'{filepath}', 'wb') as outfile:
            pickle.dump(self.tagger, outfile)

    @classmethod
    def load(self, filepath):
        '''
            Load a trained tagger for use. This method will only load
            the final tagger.
        '''
        if filepath.endswith('.pkl'):
            with open(f'{filepath}', 'rb') as infile:
                self.tagger = pickle.load(infile)
        else:
            print('Tagger error: invalid format.')
