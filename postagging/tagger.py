"""
    Module that handles the creation of several types of PoS taggers.
    Currently it supports creation of sequential backoff taggers,
    Brill taggers and Perceptron taggers. All of these taggers are
    implemented in NLTK, and this module only provides an easy way
    to work with them.
"""
import abc
import pickle
import nltk

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


class Tagger(abc.ABC):
    """
        Higher level tagger class that contains methods common to
        all types of taggers. Saved models can be loaded using this
        class.
    """

    def __init__(self, tagger=None, language='portuguese'):
        """
            Tagger constructor.

            :param tagger: tagger from NLTK
            :param language str: default language to load the tokenizer
        """
        self.tagger = tagger
        self.tokenizer = self._load_tokenizer(language)

    @staticmethod
    def _load_tokenizer(language):
        """
            Load the correct tokenizer according to the language.
            Download the punkt module if it is not present on the
            system yet.
        """
        try:
            nltk.download('punkt', quiet=True)
            tokenizer = nltk.data.load(f'tokenizers/punkt/{language}.pickle')
        except LookupError:
            raise Exception('natlang.postagging: could not find a tokenizer '
                            'for the given language')
        return tokenizer

    def evaluate(self, test):
        """
            Evaluate the trained tagger on test set and show a
            classification report.

            :param test list: list of tagged sentences (already tokenized) to
                evaluate the tagger
        """
        test_set = [[word for word, tag in sentence] for sentence in test]
        y_pred = [[tag for word, tag in self.tag(sentence)]
                  for sentence in test_set]
        y_true = [[tag for word, tag in sentence] for sentence in test]

        average_accuracy = sum(map(lambda x, y: x * len(y),
                                   map(accuracy_score, y_true, y_pred),
                                   test_set))/sum(len(x) for x in test_set)

        print(f'average tagger accuracy: {average_accuracy}')

    def save(self, filepath):
        """
            Save the current tagger to the given filepath.

            :param filepath str: output filepath for saving the tagger
        """
        tagger = Tagger(self.tagger)
        try:
            with open(f'{filepath}', 'wb') as file:
                pickle.dump(tagger, file)
        except OSError:
            raise Exception('natlang.postagging.tagger: invalid filepath')

    @staticmethod
    def load(filepath):
        """
            Load a tagger from the given filepath.

            :param filepath str: input filepath for loading the tagger
            :return: instance of Tagger class containing the loaded tagger
        """
        try:
            with open(f'{filepath}', 'rb') as file:
                tagger = pickle.load(file)
        except OSError:
            raise Exception('natlang.postagging.tagger: invalid filepath')
        return Tagger(tagger)

    def tag(self, sentence):
        """
            Tag the given tokenized sentence.

            :param sentence list: sentence to be tagged already split in
                tokens
            :return: single tagged sentence
        """
        return self.tagger.tag(sentence)

    def tag_sentences(self, sentences):
        """
            Tag list of tokenized sentences.

            :param sentences list: list of sentences (list of strings) to be
                tagged by the tagger
            :return: all sentences tagged
        """
        return [self.tag(sentence) for sentence in sentences]

    def tag_untokenized(self, sentence):
        """
            Tokenize and tag the given sentence in string format.

            :param sentence str: sentence untokenized
            :return: single tagged sentence
        """
        return self.tagger.tag(nltk.word_tokenize(sentence))

    def tag_untokenized_sentences(self, sentences):
        """
            Tokenize and tag the given list of sentences in string format.

            :param sentences list: list of strings containing untokenized
                sentences
            :return: all sentences tagged
        """
        return [self.tag_untokenized(sentence) for sentence in sentences]

    def tag_untokenized_text(self, text):
        """
            Given an entire text (multiple sentences), tokenize and tag
            each sentence.

            :param text str: untokenized text to be tagged
            :return: text split into tagged sentences
        """
        return self.tag_untokenized_sentences(self.tokenizer.tokenize(text))


class SequenceBackoffTagger(Tagger):
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
        super().__init__(tagger)
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
        if sequence is None:
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

    def save(self, filepath):
        """
            Save the current tagger to a file.

            :param filepath str: output filepath
        """
        super().save(filepath)

    @staticmethod
    def load(filepath):
        """
            Load a tagger from a file.

            :param filepath str: input filepath
            :return: Tagger instance
        """
        super().load(filepath)
