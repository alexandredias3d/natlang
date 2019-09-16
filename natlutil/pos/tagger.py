"""
    Handles the creation of several types of PoS taggers from NLTK such
    as sequential backoff, Brill and Perceptron taggers.
"""
import json
import nltk
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from itertools import chain
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from natlutil.util import flatten
from natlutil.util import UNIVERSAL_TAGSET


class TaggerBuilder:
    """
        Creates an NLTK PoS tagger using the builder design pattern for
        custom combinations of taggers.

        :param train list(list(tuple)): tagged sentences for training
        :param safe bool: flag to enable/disable safe assignment of
                                     taggers
    """

    def __init__(self, train, train_size=0.7, safe=True):
        self._prefix = f'natlutil.postagging.{self.__class__.__name__}'
        self._safe = safe
        self._train_size = train_size
        self.__tagger = None
        self.__train = train

    @property
    def _tagger(self):
        """
            Gets the tagger value (internal use only).

            :return: underlying NLTK PoS tagger
        """
        return self.__tagger

    @_tagger.setter
    def _tagger(self, tagger):
        """
            Checks if a tagger exists before setting it. Notifies of
            destructive operations if the safe flag is set to True.

            :param tagger: the tagger to be assigned to the old tagger
        """
        change = True
        msg = f'overwrites {self._tagger} by {tagger} - old tagger will be '

        if self._safe:
            if self.__tagger is not None:
                if hasattr(tagger, 'backoff'):
                    change = self._ask_confirmation(f'{msg} backoff')
                elif hasattr(tagger, '_initial_tagger'):
                    change = self._ask_confirmation(f'{msg} initial')
                else:
                    change = self._ask_confirmation(f'{msg} deleted')
        if change:
            self.__tagger = tagger

    def _ask_confirmation(self, message):
        """
            Prompts the user for confirmation before performing an assignment.

            :param message str: message to show to the user
            :return: True in case of positive answer, False otherwise
        """
        while True:
            answer = input(f'{self._prefix}: {message} ([y]/n)? ').lower()
            if answer in ['y', 'yes', '']:
                return True
            elif answer in ['n', 'no']:
                return False
            print(f'{self._prefix}: valid options are: y, yes, n, no.')

    @property
    def _train(self):
        """
            Training data getter.

            :return: tagged sentences for training
        """
        if isinstance(self.__train, (list, nltk.LazySubsequence)):
            return chain(self.__train)
        else:
            return self.__train.get_train(self._train_size)

    @_train.setter
    def _train(self, train):
        """
            Training data setter. Since most NLTK taggers require training
            data, raises an exception if it is not present.

            :param train list(list(tuple)): tagged sentences for training
        """
        if train is None:
            raise ValueError(f'{self._prefix}: train cannot be None.')
        self.__train = train

    def with_default(self, tag='NOUN'):
        """
            Creates a DefaultTagger.

            :param tag str: default tag
            :return: self reference for method chaining
        """
        self._tagger = nltk.DefaultTagger(tag)
        return self

    def with_affix(self, affix_length=3, min_stem_length=2):
        """
            Creates an AffixTagger.

            :param affix_length int: size of word affix
            :param min_stem_length int: size of word stem
            :return: self to allow method chaining
        """
        self._tagger = nltk.AffixTagger(self._train, None, affix_length,
                                        min_stem_length, self._tagger)
        return self

    def with_regex(self, regexps=None):
        """
            Creates a RegexTagger.

            :param regexps: list of tuples, each containing a regex and tag
            :return: self to allow method chaining
        """
        if regexps is None:
            regexps = [
                (r'^-?\d+(.\d+)?$', 'NUM'),
            ]

        self._tagger = nltk.RegexpTagger(regexps, self._tagger)
        return self

    def with_unigram(self):
        """
            Creates a UnigramTagger (ngram of size 1).

            :return: self reference for method chaining
        """
        self._tagger = nltk.UnigramTagger(self._train, None, self._tagger)
        return self

    def with_bigram(self):
        """
            Creates a BigramTagger (ngram of size 2).

            :return: self reference for method chaining
        """
        self._tagger = nltk.BigramTagger(self._train, None, self._tagger)
        return self

    def with_trigram(self):
        """
            Creates a TrigramTagger (ngram of size 3).

            :return: self reference for method chaining
        """
        self._tagger = nltk.TrigramTagger(self._train, None, self._tagger)
        return self

    def with_ngram(self, size=4):
        """
            Creates a NgramTagger of arbitrary size.

            :param size int: size of the ngram
            :return: self reference for method chaining
        """
        self._tagger = nltk.NgramTagger(size, self._train, None, self._tagger)
        return self

    def with_brill(self, templates=None, max_rules=200, min_score=2,
                   min_acc=None):
        """
            Creates a BrillTagger. Uses transformation based learning
            to improve performance of an initial tagger.

            :param templates:
        """
        if templates is None:
            templates = nltk.tag.brill.fntbl37()

        trainer = nltk.tag.BrillTaggerTrainer(self._tagger, templates, trace=0,
                                              deterministic=True,
                                              ruleformat='str')

        self._tagger = trainer.train(list(self._train), max_rules, min_score,
                                     min_acc)
        return self

    def with_perceptron(self, path='', iterations=5):
        """
            Creates a Perceptron tagger. If the path is an empty string,
            trains a new model. Otherwise, loads it from the given path.

            :param path str: the path to the trained model
            :param iterations int: the amount of training epochs
            :return: self reference to allow method chaining
        """
        self._tagger = nltk.PerceptronTagger(load=False)

        if not path:
            self._tagger.train(self._train, nr_iter=iterations)
        else:
            self._tagger.load(path)

        return self

    def get_result(self, lang='portuguese'):
        """
            Gets the resulting tagger embedded in a Tagger object.

            :return: the resulting NLTK tagger
        """
        return Tagger(self._tagger, lang=lang)


class Tagger:
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
        self._prefix = f'natlutil.postagging.{self.__class__.__name__}'

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
            Loads the correct tokenizer according to the language.
            Downloads punkt if it is not available locally.
        """
        try:
            nltk.download('punkt', quiet=True)
            self._sent_tokenize = nltk.data.load(
                f'tokenizers/punkt/{lang}.pickle').tokenize
        except LookupError:
            raise ValueError(f'{self._prefix}: could not find a sentence '
                             'tokenizer for the given language.')

    @staticmethod
    def _prepare_test_set(test, test_size):
        """
            Extracts features (words) and targets (tags) from the test
            set.

            :param test: the list of tagged sentences
            :return: features and targets of the given test set
        """
        if isinstance(test, list):
            iter1 = chain(test)
            iter2 = chain(test)
        else:
            iter1 = test.get_test(test_size)
            iter2 = test.get_test(test_size)

        x_feat = [[word for word, tag in sent] for sent in iter1]
        y_true = [[tag for word, tag in sent] for sent in iter2]

        return x_feat, y_true

    def evaluate(self, test, test_size=0.3, target_names=None, filename=None):
        """
            Evaluates the tagger on the given test set. The test set
            must be a list of tagged sentences.
        """
        x_feat, y_true = self._prepare_test_set(test, test_size)

        y_pred = [[tag for word, tag in sent] for sent in
                  self.tag_tokenized_sentences(x_feat)]

        y_true = flatten(y_true)
        y_pred = flatten(y_pred)

        if target_names is None:
            target_names = UNIVERSAL_TAGSET

        accuracy = self.compute_accuracy(y_true, y_pred)

        conf_matrix = self.compute_confusion_matrix(y_true, y_pred,
                                                    labels=target_names)

        conf_matrix = np.nan_to_num(conf_matrix)

        report = self.compute_classification_report(y_true, y_pred,
                                                    target_names,
                                                    output_dict=True)

        self._plot(conf_matrix, report, target_names, filename)

        return accuracy, conf_matrix, report

    def _plot(self, conf_matrix, report, target_names, filename):
        grid = (1, 3)

        fig = plt.figure()

        ax1 = plt.subplot2grid(grid, (0, 0), colspan=2, rowspan=1, fig=fig)
        ax2 = plt.subplot2grid(grid, (0, 2), colspan=1, rowspan=1, fig=fig)

        sns.set()
        self._plot_confusion_matrix(ax1, conf_matrix, target_names)
        self._plot_classification_report(ax2, report, target_names)

        dpi = 1200
        fig.set_size_inches(1920/fig.dpi, 1080/fig.dpi)
        fig.savefig(filename, format='pdf', bbox_inches='tight', dpi=dpi)


    @staticmethod
    def _plot_confusion_matrix(ax, conf_matrix, target_names, normalize=True,
                               decimals=2):
        fmt = f'.{decimals}f' if normalize else 'd'
        vmin = 0.0
        vmax = 1.0 if normalize else conf_matrix.max()
        cmap = 'Greens'

        sns.heatmap(conf_matrix, vmin, vmax, cmap, annot=True, fmt=fmt,
                    square=True, yticklabels=target_names, ax=ax)

        xlim = ax.get_xlim()
        ax.set_ylim(xlim[1], xlim[0])

        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        ax.set_xticklabels(target_names, rotation=45)
        ax.set_title('Confusion Matrix' + (' Normalized' if normalize else ''))

    @staticmethod
    def _plot_classification_report(ax, report, target_names, decimals=2):
        unused = ['accuracy', 'micro avg', 'macro avg', 'weighted avg']

        cell_values = []
        for key, subreport in report.items():
            if key not in unused:
                cell_values.append([f'{value:.{decimals}f}'
                                    for value in subreport.values()])

        metrics = ['precision', 'recall', 'f1-score', 'support']
        table = ax.table(cellText=cell_values,
                         rowLabels=target_names,
                         colLabels=metrics,
                         loc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(13)
        ax.axis('off')

    @staticmethod
    def compute_accuracy(y_true, y_pred, normalize=True, decimals=2):
        """
            Computes the tagger accuracy using sklearn.

            :param y_true: the ground truth tags
            :param y_pred: the predicted tags
        """
        return round(accuracy_score(y_true, y_pred, normalize), decimals)

    @staticmethod
    def compute_confusion_matrix(y_true, y_pred, normalize=True, labels=None,
                                 decimals=2):
        """
            Computes the confusion matrix using sklearn.

            :param y_true: the ground truth tags
            :param y_pred: the predicted tags
            :return: confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        if normalize:
            cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
            cm = np.around(cm, decimals=decimals)

        return cm

    @staticmethod
    def compute_classification_report(y_true, y_pred, target_names=None,
                                      output_dict=False):
        """
            Computes the precision, recall, f1-score and support for
            each tag using sklearn.

            :param y_true: the ground truth tags
            :param y_pred: the predicted tags
            :param target_names: the list of label names to use in the
                                 report
            :param output_dict: True if should return a dict, False if
                                it should return a string
            :return: report in string or dict format
        """
        report = classification_report(y_true, y_pred,
                                       labels=target_names,
                                       output_dict=output_dict)
        return report

    def save(self, filepath):
        """
            Saves the current tagger to the given filepath.

            :param filepath str: the output tagger filepath
        """
        try:
            with open(f'{filepath}', 'w') as file:
                json.dump(self.tagger, file, indent=1,
                          cls=nltk.JSONTaggedEncoder)
        except OSError:
            raise Exception(f'{self._prefix}: invalid filepath')

    @classmethod
    def load(cls, filepath):
        """
            Loads a tagger from the given filepath.

            :param filepath str: the input tagger filepath
            :return: instance of Tagger with the loaded tagger
        """
        try:
            with open(f'{filepath}', 'r') as file:
                tagger = json.load(file, cls=nltk.JSONTaggedDecoder)
        except OSError:
            raise Exception(
                f'natlutil.postagging.{cls.__name__}: invalid filepath')
        return cls(tagger)

    def tag(self, arg):
        """
            Checks the given argument and call the correct tagging for its
            type. If performance is critical, use the specific methods
            instead of this one.

            :param arg: the sentence, list of sentences or text for tagging
            :return: the sentence(s) of arg tagged
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
        raise ValueError(f'{self._prefix}: invalid argument format.')

    def tag_tokenized(self, sentence):
        """
            Tags the given tokenized sentence (single list of strings).

            :param sentence list: the tokenized sentence to be tagged
            :return: single tagged sentence
        """
        return self.tagger.tag(sentence)

    def tag_tokenized_sentences(self, sentences):
        """
            Tags the given tokenized sentences (multiple list of
            strings).

            :param sentences list: the tokenized sentences to be tagged
            :return: all sentences tagged
        """
        return [self.tag_tokenized(sentence) for sentence in sentences]

    def tag_untokenized(self, sentence):
        """
            Tags a single untokenized sentence (a single string).

            :param sentence str: the untokenized sentence to be tagged
            :return: single tagged sentence
        """
        return self.tagger.tag(self._word_tokenize(sentence))

    def tag_untokenized_sentences(self, sentences):
        """
            Tags the given untokenized sentences (single list of strings).

            :param sentences list: the untokenized sentences to be
                                   tagged
            :return: all sentences tagged
        """
        return [self.tag_untokenized(sentence) for sentence in sentences]

    def tag_untokenized_text(self, text):
        """
            Tags the given untokenized text (a single string).

            :param text str: the untokenized text to be tagged
            :return: text split into tagged sentences
        """
        return self.tag_untokenized_sentences(self._sent_tokenize(text))
