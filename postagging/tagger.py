import pickle
import nltk


class SequenceBackoffTagger:
    '''
        Manage creation of a sequence of backoff taggers.
    '''

    def __init__(self, sequence=None, reverse=True, train=None, tag=None,
                 n=None, regexps=None, affix_length=None,
                 min_stem_length=None):
        '''
            Class constructor. Receives a sequence from the
            sequence_backoff_factory method and creates the
            given tagger combination.
        '''
        self._defaults = self._get_taggers_args(train, tag, n, affix_length,
                                                min_stem_length, regexps)

        taggers = []
        self._validate_sequence(sequence)
        self._validate_tagger_names(sequence)
        sequence = reversed(sequence) if reverse else sequence

        # Used cls here to give notion that it is in fact instantiating a class
        for cls in sequence:
            if taggers:
                self._defaults['backoff'] = taggers[-1]
                print(self._defaults)
            cls_args = self._inspect_tagger_constructor_args(cls)
            taggers.append(cls(**cls_args))

        self.tagger = taggers[-1]

    @classmethod
    def _validate_sequence(cls, sequence):
        '''
            Check if the sequence is a list of strings.
        '''
        if not all(isinstance(element, str) for element in sequence):
            raise Exception('natlang.postagging: sequence must be a list of'
                            'strings')

    @classmethod
    def _validate_tagger_names(cls, sequence):
        '''
            Map string names into NLTK tagger classes.
        '''
        mapping = {'DEFAULT': nltk.DefaultTagger,
                   'UNIGRAM': nltk.UnigramTagger,
                   'BIGRAM':  nltk.BigramTagger,
                   'TRIGRAM': nltk.TrigramTagger,
                   'NGRAM':   nltk.NgramTagger,
                   'AFFIX':   nltk.AffixTagger,
                   'REGEX':   nltk.RegexpTagger,
                   'REGEXP':  nltk.RegexpTagger}

        def _clean(name):
            return name.upper().replace('TAGGER', '').replace(' ', '')

        try:
            for idx, tagger in enumerate(sequence):
                sequence[idx] = mapping[_clean(tagger)]
        except KeyError:
            raise Exception('natlang.postagging: invalid tagger name')

    def _inspect_tagger_constructor_args(self, cls):
        '''
            Given a class and the dictionary containing the defaults,
            return the defaults that the class should use.

            Notes:
                - co_argcount: number of arguments
                - co_varnames: local variable names, starting by arguments
                - first argument is always self
        '''
        argcount = cls.__init__.__code__.co_argcount
        varnames = cls.__init__.__code__.co_varnames

        return {key: self._defaults[key] for key in varnames[1:argcount] if
                key in self._defaults}

    @classmethod
    def _get_taggers_args(cls, train, tag, n, affix_length,
                          min_stem_length, regexps):
        '''
            Set the default values to be used as arguments on tagger
            creation.
        '''
        defaults = {
            'train': cls._get_train(train),
            'tag': cls._get_default_tag(tag),
            'n': cls._get_ngram_n(n),
            'affix_length': cls._get_affix_affix_length(affix_length),
            'min_stem_length': cls._get_affix_mim_stem_length(min_stem_length),
            'regexps': cls._get_regexp_regexps(regexps)
                    }
        return defaults

    @classmethod
    def _get_train(cls, train):
        '''
            Given a train set as a parameter, check if it is not None.
            Raise an error in case the train is None (some Sequential
            Backoff Taggers need to train on data).
        '''
        if train is not None:
            return train
        raise Exception('natlang.postagging: missing train data for '
                        'SequenceBackoffTagger')

    @classmethod
    def _get_default_tag(cls, tag):
        '''
            Get the tag to be used by the Default Tagger. If tag is None,
            the default tag is 'N' (classify all words as nouns).
        '''
        return tag if tag else 'N'

    @classmethod
    def _get_ngram_n(cls, n):
        '''
            Get the size of the Ngram Tagger. Currently it supports only one
            Ngram tagger. If n is None, the default size n is 4.
        '''
        return n if n is not None else 4

    @classmethod
    def _get_affix_affix_length(cls, affix_length):
        '''
            Get the size of the affix to be used by the Affix Tagger. If
            affix_length is None, the default length is 3.
        '''
        return affix_length if affix_length else 3

    @classmethod
    def _get_affix_mim_stem_length(cls, min_stem_length):
        '''
            Get the minimum size of the stem to be used by the Affix Tagger. If
            min_stem_length is None, the default length is 2.
        '''
        return min_stem_length if min_stem_length else 2

    @classmethod
    def _get_regexp_regexps(cls, regexps):
        '''
            Get the regex patterns to be used by the RegexP Tagger. If regexps
            is None, the default regex patterns are defined in the list
            patterns below.
        '''
        patterns = []
        return regexps if regexps else patterns

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
        # headers = ['Tagger', 'Accuracy']
        # row = '{:<10}' * len(headers)
        # print(row.format(*headers))
        # row = '{:<10}{:<10.4f}'
        # for tagger in self.taggers:
        #     print(row.format(tagger.__class__.__name__.replace('Tagger', ''),
        #                      tagger.evaluate(test_set)))

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
    def load(cls, filepath):
        '''
            Load a trained tagger for use. This method will only load
            the final tagger.
        '''
        if filepath.endswith('.pkl'):
            with open(f'{filepath}', 'rb') as infile:
                tagger = pickle.load(infile)
        else:
            print('Tagger error: invalid format.')
            tagger = None

        return tagger
