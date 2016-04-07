import logging

import numpy as np
import scipy as sp


class Identity(object):  # IStandardizer #!!LATER make an abstract object

    """Do nothing to the data"""

    def __init__(self):
        pass

    def standardize(self, snps, blocksize=None, force_python_only=False):
        return snps

    def __repr__(self):
        return "{0}()".format(self.__class__.__name__)
