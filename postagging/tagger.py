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

from util import flatten

class Tagger(abc.ABC):
    """
        Higher level tagger class that contains methods common to
        all types of taggers. Saved models can be loaded using this
        class.
    """

    def __init__(self, tagger=None, lang='portuguese'):
        """
            Tagger constructor.

            :param tagger: tagger from NLTK
            :param lang str: default language to load the tokenizer
        """
        self.tagger = tagger
        self._sent_tokenizer = None
        self._sent_tokenize = None
        self._word_tokenize = nltk.word_tokenize

        self._load_sent_tokenizer(lang)

    @property
    def tagger(self):
        """
            Tagger gettter.
        """
        return self._tagger

    @tagger.setter
    def tagger(self, value):
        """
            Tagger setter.

            :param value: tagger object from NLTK
        """
        self._tagger = value

    def _load_sent_tokenizer(self, lang):
        """
            Load the correct tokenizer according to the language.
            Download the punkt module if it is not present on the
            system yet.
        """
        try:
            nltk.download('punkt', quiet=True)
            self._sent_tokenizer = nltk.data.load(f'tokenizers/punkt/'
                                                  f'{lang}.pickle')
            self._sent_tokenize = self._sent_tokenizer.tokenize
        except LookupError:
            raise ValueError('natlang.postagging: could not find a sentence '
                             'tokenizer for the given language')

    def accuracy(self, y_true, y_pred, normalize=True):
        """
            Compute the tagger accuracy. Flatten lists to calculate
            the accuracy. It is also possible to achieve the same
            result using map, sum, and lambdas to compute the
            weighted average.

            :param y_true: ground truth
            :param y_pred: predicted values by tagger
        """

        y_true_flat = flatten(y_true)
        y_pred_flat = flatten(y_pred)

        return accuracy_score(y_true_flat, y_pred_flat, normalize)

    def confusion_matrix(self):
        pass

    def report(self):
        pass

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

    def tag(self, arg):
        """
            Call the correct tag method based on the given input. If
            performance is critical, call the correct method instead of
            this one. In the worst case scenario, this method might check
            all elements on arg twice (in the case arg contains a list).

            :param arg: sentence, list of sentences or text for tagging
        """
        if isinstance(arg, str):
            return self.tag_untokenized_text(arg)
        elif isinstance(arg, list):
            if all(isinstance(item, str) for item in arg):
                if all(len(item) == 1 for item in arg):
                    return self.tag_tokenized(arg)
                else:
                    print('Here')
                    return self.tag_untokenized_sentences(arg)
            elif all(isinstance(item, list) for item in arg):
                return self.tag_tokenized_sentences(arg)
        raise ValueError('natlang.postagging: invalid argument format')

    def tag_tokenized(self, sentence):
        """
            Tag the given tokenized sentence.

            :param sentence list: sentence to be tagged already
                split in tokens
            :return: single tagged sentence
        """
        return self.tagger.tag(sentence)

    def tag_tokenized_sentences(self, sentences):
        """
            Tag list of tokenized sentences.

            :param sentences list: list of sentences (list of strings) to be
                tagged by the tagger
            :return: all sentences tagged
        """
        return [self.tag_tokenized(sentence) for sentence in sentences]

    def tag_untokenized(self, sentence):
        """
            Tokenize and tag the given sentence in string format.

            :param sentence str: sentence untokenized
            :return: single tagged sentence
        """
        return self.tagger.tag(self._word_tokenize(sentence))

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
        return self.tag_untokenized_sentences(self._sent_tokenize(text))


class SequenceBackoffTagger(Tagger):
    """
        Manage creation of a sequence of backoff taggers.
    """

    def __init__(self, sequence=None, reverse=True, tagger=None,
                 train=None, default_tag=None, ngram_size=None,
                 regexps=None, affix_length=None, min_stem_length=None):
        """
            Class constructor. Receives a sequence from the
            sequence_backoff_factory method and creates the
            given tagger combination.
        """
        if tagger is not None:
            super().__init__(tagger)
        else:
            self.sequence = sequence
            self.tagger = tagger
            self.train = train
            self.ngram_size = ngram_size
            self.default_tag = default_tag
            self.regexps = regexps
            self.affix_length = affix_length
            self.min_stem_length = min_stem_length

            self.defaults = {
                    'train':            self.train,
                    'tag':              self.default_tag,
                    'n':                self.ngram_size,
                    'affix_length':     self.affix_length,
                    'min_stem_length':  self.min_stem_length,
                    'regexps':          self.regexps
                }
            super().__init__(self._create_tagger_sequence(reverse))

    @property
    def sequence(self):
        """
            Sequence getter.
        """
        return self._sequence

    @sequence.setter
    def sequence(self, value):
        """
            Sequence setter. Set sequence to the given value if it is
            a list of strings. Raise an exception if at least one element
            of the list is not a string. If value is None, set sequence
            to the default value:
            ['trigram', 'bigram', 'unigram', 'regex', 'affix', 'default']

            :param value list: sequence of tagger names
        """
        if value is None:
            self._sequence = ['trigram', 'bigram', 'unigram', 'regex',
                              'affix', 'default']
        elif all(isinstance(element, str) for element in value):
            self._sequence = value
        else:
            raise ValueError('natlang.postagging: sequence must be a list of '
                             'strings containing tagger names')

    def _tagger_names(self, reverse):
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
            taggers = []
            for tagger in self.sequence:
                taggers.append(mapping[_clean(tagger)])
            return reversed(taggers) if reverse else taggers
        except KeyError:
            raise Exception('natlang.postagging: invalid tagger name')

    def _create_tagger_sequence(self, reverse):
        """
            Create sequence of taggers an return the most external one.
        """
        taggers = []
        for cls in self._tagger_names(reverse):
            if taggers:
                self.defaults['backoff'] = taggers[-1]
            cls_args = self._inspect_tagger_constructor_args(cls)
            taggers.append(cls(**cls_args))

        return taggers[-1]

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

        return {key: self.defaults[key] for key in varnames[1:argcount] if
                key in self.defaults}

    @property
    def train(self):
        """
            Training data getter.
        """
        return self._train

    @train.setter
    def train(self, value):
        """
            Training data setter. Set training data to value if it is not
            None, raise an exception otherwise (some taggers need training
            data).

            :param value list: list of sentences for training
        """
        if value is not None:
            self._train = value
        else:
            raise Exception('natlang.postagging: missing training data for '
                            'SequenceBackoffTagger')

    @property
    def default_tag(self):
        """
            Default tag getter.
        """
        return self._default_tag

    @default_tag.setter
    def default_tag(self, value):
        """
            Default tag setter. Set the default tag to value if it is not
            None, set to 'N' otherwise.

            :param value str: default tag
        """
        self._default_tag = value if value else 'N'

    @property
    def ngram_size(self):
        """
            N-gram size getter.
        """
        self._ngram_size

    @ngram_size.setter
    def ngram_size(self, value):
        """
            N-gram size setter. Set the n-gram (contiguous sequence of size n)
            size to value if it is not None, set to 4 otherwise.

            :param value int: size of ngram
        """
        self._ngram_size = value if value else 4

    @property
    def affix_length(self):
        """
            Affix length getter.
        """
        return self._affix_length

    @affix_length.setter
    def affix_length(self, value):
        """
            Affix length setter. Set affix length of word to value if it is
            not None, set to 3 otherwise.

            :param value int: size of word affix
        """
        self._affix_length = value if value else 3

    @property
    def min_stem_length(self):
        """
            Minimum stem length getter.
        """
        return self._min_stem_length

    @min_stem_length.setter
    def min_stem_length(self, value):
        """
            Minimum stem length setter. Set minimum stem length of word to
            value if it is not None, set to 2 otherwise.

            :param value int: size of word stem
        """
        self._min_stem_length = value if value else 2

    @property
    def regexps(self):
        """
            Regex patterns getter.
        """
        return self._regexps

    @regexps.setter
    def regexps(self, value):
        """
            Regex patterns setter. Set the regex patterns to value if it is
            not None, set to local variable patterns otherwise.

            :param value list: list of regular expressions
        """
        patterns = [
            (r'^-?\d+(.\d+)?$', 'NUM')
        ]
        self._regexps = value if value else patterns
