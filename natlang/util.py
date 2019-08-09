"""
    Provides utility functions for the PoS tagging classes.
"""
from itertools import chain

UNIVERSAL_TAGSET = ['VERB', 'NOUN', 'PRON', 'ADJ', 'ADV', 'ADP', 'CONJ',
                    'DET', 'NUM', 'PRT', 'X', '.']


def flatten(nonflat):
    """
        Reduces one level of the given nested list.

        :param nonflat list: nonflattened list
        :return: flattened list
    """
    return list(chain.from_iterable(nonflat))
