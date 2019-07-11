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
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from util import flatten
from util import UNIVERSAL_TAGSET


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
        self._sent_tokenize = None
        self._word_tokenize = nltk.word_tokenize

        self._load_sent_tokenizer(lang)

        print(self._sent_tokenize, self._word_tokenize)

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
            self._sent_tokenize = nltk.data.load(f'tokenizers/punkt/'
                                                 f'{lang}.pickle').tokenize
        except LookupError:
            raise ValueError('natlang.postagging: could not find a sentence '
                             'tokenizer for the given language')

    @staticmethod
    def _prepare_test_set(test):
        """
            Extract features (words in sentences) and targets (tags) from
            the test set.

            :param test: list of tagged sentences
            :return: features and targets of the given test set
        """
        x_feat = [[word for word, tag in sent] for sent in test]
        y_true = [[tag for word, tag in sent] for sent in test]

        return x_feat, y_true

    def evaluate(self, test, accuracy=True, conf_matrix=True,
                 plot=False, report=True, normalize=True,
                 target_names=None, output_dict=True, decimals=2):
        """
            Evaluate the tagger on the given test set. The test set
            must be a list of tagged sentences.
        """
        x_feat, y_true = self._prepare_test_set(test)

        y_pred = [[tag for word, tag in sent] for sent in
                  self.tag_tokenized_sentences(x_feat)]

        y_true = flatten(y_true)
        y_pred = flatten(y_pred)

        if target_names is None:
            target_names = UNIVERSAL_TAGSET

        if accuracy:
            accuracy = self._accuracy(y_true, y_pred, normalize, decimals)
        if conf_matrix:
            conf_matrix = self._confusion_matrix(y_true, y_pred, normalize,
                                                 plot, labels=target_names,
                                                 decimals=decimals)
        if report:
            report = self._classification_report(y_true, y_pred,
                                                 target_names,
                                                 output_dict)
        if plot:
            fmt = f'.{decimals}f' if normalize else 'd'
            grid = (3, 6)

            ax1 = plt.subplot2grid(grid, (0, 0), colspan=4, rowspan=3)
            ax2 = plt.subplot2grid(grid, (0, 4), colspan=2, rowspan=3)

            sns.set()
            sns.heatmap(conf_matrix, cmap='Greens', annot=True, robust=True,
                        fmt=fmt, xticklabels=target_names,
                        yticklabels=target_names, ax=ax1)
            ax1.set_ylabel('True label')
            ax1.set_xlabel('Predicted label')
            ax1.set_title('Confusion Matrix '
                          + ('Normalized' if normalize else ''))

            unused = ['accuracy', 'macro avg', 'weighted avg']

            cell_values = []
            for key, subreport in report.items():
                if key not in unused:
                    cell_values.append([f'{value:.{decimals}f}'
                                        for value in subreport.values()])

            metrics = ['precision', 'recall', 'f1-score', 'support']
            table = ax2.table(cellText=cell_values,
                              rowLabels=target_names,
                              colLabels=metrics,
                              loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(13)
            ax2.axis('off')

            manager = plt.get_current_fig_manager()
            manager.resize(*manager.window.maxsize())

            plt.show()

        return accuracy, conf_matrix, report

    @staticmethod
    def _accuracy(y_true, y_pred, normalize=True, decimals=2):
        """
            Compute the tagger accuracy using sklearn. Flatten
            lists to calculate the accuracy. It is also possible
            to achieve the same result using map, sum, and lambdas
            to compute the weighted average.

            :param y_true: ground truth tags
            :param y_pred: predicted tags by the tagger
        """
        return round(accuracy_score(y_true, y_pred, normalize), decimals)

    @staticmethod
    def _confusion_matrix(y_true, y_pred, normalize=True, plot=False,
                          labels=None, decimals=2):
        """
            Use sklearn to compute the confusion matrix. Plot the
            figure if plot is True.

            :param y_true: ground truth tags
            :param y_pred: predicted tags by the tagger
            :return: confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
            cm = np.around(cm, decimals=decimals)

        return cm

    @staticmethod
    def _classification_report(y_true, y_pred,
                               target_names=None,
                               output_dict=False):
        """
            Use sklearn to compute the classification report, in which
            the precision, recall, f1-score and support are calculated
            for each tag.

            :param y_true: ground truth tags
            :param y_pred: predicted tags by the tagger
            :param target_names: list of label names to be used in the
                report
            :param output_dict: True if should return a dict, False if
                it should return a string
            :return: report in string or dict format
        """
        report = classification_report(y_true, y_pred,
                                       target_names=target_names,
                                       output_dict=output_dict)
        return report

    def save(self, filepath):
        """
            Save the current tagger to the given filepath.

            :param filepath str: output filepath for saving the tagger
        """
        try:
            with open(f'{filepath}', 'wb') as file:
                pickle.dump(self, file)
        except OSError:
            raise Exception('natlang.postagging.tagger: invalid filepath')

    @classmethod
    def load(cls, filepath):
        """
            Load a tagger from the given filepath.

            :param filepath str: input filepath for loading the tagger
            :return: instance of Tagger class containing the loaded tagger
        """
        try:
            with open(f'{filepath}', 'rb') as file:
                tagger = pickle.load(file)
        except OSError:
            raise Exception(f'natlang.postagging.__class__.__name__: invalid '
                            f'filepath')
        return tagger

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

    def __init__(self, sequence, reverse=True, tagger=None,
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
            None, set to 'NOUN' otherwise.

            :param value str: default tag
        """
        self._default_tag = value if value else 'NOUN'

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


class BrillTagger(Tagger):
    """
        Manage the creation of a Brill Tagger, which, given an baseline
        tagger, is capable of improving its performance through
        transformation based learning.
    """

    def __init__(self, initial_tagger, train, templates=None,
                 trace=0, deterministic=None, ruleformat='str',
                 max_rules=200, min_score=2, min_acc=None):

        self.initial_tagger = initial_tagger
        self.train = train
        self.templates = templates
        self.trace = trace
        self.deterministic = deterministic
        self.ruleformat = ruleformat
        self.max_rules = max_rules
        self.min_score = min_score
        self.min_acc = min_acc

        self._train_brill()

    @property
    def initial_tagger(self):
        """
            Initial tagger getter.
        """
        return self._initial_tagger

    @initial_tagger.setter
    def initial_tagger(self, initial_tagger):
        """
            Initial tagger setter. The Brill Tagger improves the performance
            of the given tagger through transitional based learning, where
            patches are applied to rules aiming to reduce the error.

            :param initial_tagger: baseline tagger
        """
        if initial_tagger is not None:
            self._initial_tagger = initial_tagger
        else:
            raise Exception('natlang.postagging: missing initial tagger for '
                            'BrillTagger')

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
            None, raise an exception otherwise (Brill Tagger needs training
            data to improve the performance of the given initial tagger).

            TODO: consider moving the training property to the super class.

            :param value list: list of sentences for training
        """
        if value is not None:
            self._train = value
        else:
            raise Exception('natlang.postagging: missing training data for '
                            'BrillTagger')

    @property
    def templates(self):
        """
            Templates getter.
        """
        return self._templates

    @templates.setter
    def templates(self, templates):
        """
            Templates setter. The Brill Tagger rules are built upon templates.
            Each template has a combination of the surrounding features (words
            and PoS tags).

            By default it uses 37 templates based on the fnTBL.
        """
        self._templates = templates if templates else nltk.tag.brill.fntbl37()

    @property
    def trace(self):
        """
            Trace getter.
        """
        return self._trace

    @trace.setter
    def trace(self, trace):
        """
            Trace setter. The verbosity of the Brill Tagger training process.
            Higher values means that more information will be printed. After
            looking at the BrillTaggerTrainer code, it seems that 4 is the
            biggest possible value.
        """
        self._trace = trace if trace else 0

    @property
    def deterministic(self):
        """
            Tie breaking getter.
        """
        return self._deterministic

    @deterministic.setter
    def deterministic(self, deterministic):
        """
            Tie breaking setter. Defines if the tie breaking should be
            deterministic (True) or not (False). By default, it uses
            deterministic tie breaking to provide consistency between
            different runs.
        """
        self._deterministic = deterministic if deterministic else True

    @property
    def ruleformat(self):
        """
            Rule format getter.
        """
        return self._ruleformat

    @ruleformat.setter
    def ruleformat(self, ruleformat):
        """
            Rule format setter. Format that should be use when outputing a
            rule.
        """
        self._ruleformat = ruleformat if ruleformat else 'str'

    @property
    def max_rules(self):
        """
            Maximum rules getter.
        """
        return self._max_rules

    @max_rules.setter
    def max_rules(self, max_rules):
        """
            Maximum rules setter. The tranining process will produce at most
            the value defined in maximum rules.
        """
        self._max_rules = max_rules if max_rules else 200

    @property
    def min_score(self):
        """
            Mininimum score getter.
        """
        return self._min_score

    @min_score.setter
    def min_score(self, min_score):
        """
            Minimum score setter. The score threshold that will be used to
            decide if a rule should be modified or not.
        """
        self._min_score = min_score if min_score else 2

    @property
    def min_acc(self, ):
        """
            Mininimum accuracy getter.
        """
        return self._min_acc

    @min_acc.setter
    def min_acc(self, min_acc):
        """
            Minimum accuraccy setter. The accuracy threshold that will be used
            to decide if a rule should be modified or not.
        """
        self._min_acc = min_acc if min_acc else None

    def _train_brill(self):
        """
            Create a Brill Tagger Trainer and train a Brill Tagger using the
            training process.
        """
        trainer = nltk.tag.BrillTaggerTrainer(self.initial_tagger,
                                              self.templates,
                                              self.trace,
                                              self.deterministic,
                                              self.ruleformat)

        self.tagger = trainer.train(self.train, self.max_rules,
                                    self.min_score, self.min_acc)

    def print_rules(self):
        """
            Exhibit the rules obtained by the Brill Tagger in the
            training process.
        """
        print(self.tagger.rules)

    def print_templates(self):
        """
            Exhibit the templates currently being used by the Brill
            Tagger after the training process.
        """
        self.tagger.print_template_statistics()
