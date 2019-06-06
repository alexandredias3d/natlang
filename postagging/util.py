"""
    Provide utility functions for the PoS tagging classes.
"""
from itertools import chain

def flatten(nonflat):
    """
        Reduce one level of the given nested list.

        :param nonflat list: nonflat list
        :return: flat list
    """
    return chain.from_iterable(nonflat)
