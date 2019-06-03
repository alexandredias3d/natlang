"""
    Module that handles the creation of several types of PoS taggers.
    Currently it supports creation of sequential backoff taggers,
    Brill taggers and Perceptron taggers. All of these taggers are
    implemented in NLTK, and this module only provides an easy way
    to work with them.
"""

import pickle
import nltk

from sklearn.metrics import classification_report


class SequenceBackoffTagger:
    """
        Manage creation of a sequence of backoff taggers.
    """

    def __init__(self, tagger=None, sequence=None, reverse=True,
                 train=None, tag=None, n=None, regexps=None,
                 affix_length=None, min_stem_length=None):
        """
            Class constructor. Receives a sequence from the
            sequence_backoff_factory method and creates the
            given tagger combination.
        """
        if tagger:
            self.tagger = tagger
        else:
            self._defaults = self._get_taggers_args(train, tag, n,
                                                    affix_length,
                                                    min_stem_length, regexps)
            taggers = []
            sequence = self._validate_sequence(sequence)
            self._validate_tagger_names(sequence)
            sequence = reversed(sequence) if reverse else sequence

            for cls in sequence:
                if taggers:
                    self._defaults['backoff'] = taggers[-1]
                cls_args = self._inspect_tagger_constructor_args(cls)
                taggers.append(cls(**cls_args))

            self.tagger = taggers[-1]

    @staticmethod
    def _validate_sequence(sequence):
        """
            Check if the sequence is a list of strings.
        """
        if sequence is not None:
            return ['Trigram', 'Bigram', 'Regex', 'Unigram', 'Affix',
                    'Default']

        if all(isinstance(element, str) for element in sequence):
            return sequence

        raise Exception('natlang.postagging: sequence must be a list of'
                        'strings')

    @staticmethod
    def _validate_tagger_names(sequence):
        """
            Map string names into NLTK tagger classes.
        """
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
        """
            Given a class and the dictionary containing the defaults,
            return the defaults that the class should use.

            :param cls: tagger class
            :return: dictionary wih default values for arguments of the
                taggers' constructor
        """
        # Notes:
        #   - co_argcount: number of arguments
        #   - co_varnames: local variable names, starting by arguments
        #   - first argument is always self
        argcount = cls.__init__.__code__.co_argcount
        varnames = cls.__init__.__code__.co_varnames

        return {key: self._defaults[key] for key in varnames[1:argcount] if
                key in self._defaults}

    def _get_taggers_args(self, train, tag, n, affix_length,
                          min_stem_length, regexps):
        """
            Set the default values to be used as arguments on tagger
            creation.

            :param train list: list of sentences for training
            :param tag tuple: tuple containing word and PoS tag
            :param n int: size of ngram
            :param affix_length int: size of word affix
            :param min_stem_length int: minimum size of word stem
            :param regexps list: list of regular expressions
        """
        defaults = {
            'train': self._get_train(train),
            'tag': self._get_default_tag(tag),
            'n': self._get_ngram_n(n),
            'affix_length': self._get_affix_affix_length(affix_length),
            'min_stem_length': self._get_affix_mim_stem_length(
                min_stem_length),
            'regexps': self._get_regexp_regexps(regexps)
        }
        return defaults

    @staticmethod
    def _get_train(train):
        """
            Given a trainining set as a parameter, check if it is not None.
            Raise an error in case the train is None (some Sequential
            Backoff Taggers need to train on data).

            :param train list: list of sentences for training
            :return: train if it is not None or raise and exception otherwise
        """
        if train is not None:
            return train
        raise Exception('natlang.postagging: missing train data for '
                        'SequenceBackoffTagger')

    @staticmethod
    def _get_default_tag(tag):
        """
            Get the tag to be used by the Default Tagger.

            :param tag str: default tag
            :return: tag if it is not None, 'N' otherwise (universal tag
                for Noun)
        """
        return tag if tag else 'N'

    @staticmethod
    def _get_ngram_n(n):
        """
            Get the size of the Ngram Tagger. Currently it supports only one
            Ngram tagger.

            :param n int: size of ngram
            :return: n if it is not None, 4 otherwise
        """
        return n if n is not None else 4

    @staticmethod
    def _get_affix_affix_length(affix_length):
        """
        Get the size of the affix to be used by the Affix Tagger.

            :param affix_length int: size of word affix
            :return: affix_length if it is not None, 3 otherwise
        """
        return affix_length if affix_length else 3

    @staticmethod
    def _get_affix_mim_stem_length(min_stem_length):
        """
            Get the minimum size of the stem to be used by the Affix Tagger.

            :param min_stem_length int: size of word stem
            :return: return min_stem_length if it is not None, 2 otherwise
        """
        return min_stem_length if min_stem_length else 2

    @staticmethod
    def _get_regexp_regexps(regexps):
        """
            Get the regex patterns to be used by the RegexP Tagger.

            :param regexps list: list of regular expressions
            :return: regexps if it is not None, patterns otherwise
        """
        patterns = [
            (r'^-?\d+(.\d+)?$', 'NUM')
        ]
        return regexps if regexps else patterns

    def tag_sentence(self, sentence):
        '''
            Tag the given sentence. If sentence is a string, it will
            tokenize its words first.
        '''
        if isinstance(sentence, str):
            sentence = nltk.word_tokenize(sentence)

        return self.tagger.tag(sentence)

    def tag_sentences(self, sentences):
        '''
            Tag the given sentences.
        '''
        return [self.tag_sentence(sentence) for sentence in sentences]

    def evaluate(self, test, universal=True):
        '''
            Evaluate the trained tagger on test set and present a
            classification report.
        '''
        test_set = [[word for word, tag in sentence] for sentence in test]
        test_predicted = [self.tag_sentence(sentence) for sentence in test_set]

        def _post_process(unprocessed):
            processed = []
            for sublist in unprocessed:
                for w, t in sublist:
                    processed.append(t)
            return processed

        print(classification_report(_post_process(test),
                                    _post_process(test_predicted)))

    def save(self, filepath):
        """
            Save a trained tagger for future use. This method will only
            save the final tagger.

            :param filepath str: filepath to save the trained tagger
        """
        if not filepath.endswith('.pkl'):
            filepath += '.pkl'
        with open(f'{filepath}', 'wb') as outfile:
            pickle.dump(self.tagger, outfile)

    @staticmethod
    def load(filepath):
        """
            Load a trained tagger for use. This method will only load
            the final tagger.

            :param filepath str: filepath to load the trained tagger
            :return: the read tagger as an NLTK tagger
        """
        if filepath.endswith('.pkl'):
            with open(f'{filepath}', 'rb') as infile:
                tagger = pickle.load(infile)
        else:
            print('natlang.postagging: invalid tagger format')
            tagger = None

        return tagger
