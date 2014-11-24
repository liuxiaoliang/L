#! /usr/bin/env python
# -*- encoding: utf-8 -*- 
#
# Copyright (c) <2014> <leon.max.liew@gmail.com>
#

"""pca

principal component analysis.
"""

__author__ = 'xiaoliang liu'

import os
import sys
sys.path.append("../../common/")
sys.path.append("../linalg")
from svd import SVD
import numpy as np

class PCABase(object):
    """pca base

    pX = Y: p is loading matrix and Y is called scores.
    X, Y: column is the sample, row is the  variable.
    """
    def __init__(self, arr):
        self.arr = arr
        self.var_num = arr.shape[0] # row is the  variable
        self.loadings = None
        self.explained_var = None

    def pca(self):
        """
        
        Returns: loading matrix and explained_var.
        """
        return None
    

class PCABySVD(PCABase):
    """pca by svd

    """
    def __init__(self, arr):
        PCABase.__init__(self, arr)

    def pca(self):
        svd = SVD(arr)
        u, s, vt = svd.svd()
        variances = np.diag(s)**2
        variances_sum = sum(variances)
        self.explained_var = variances / variances_sum
        self.loadings = u.T


class PCAByNipals(PCABase):
    """pca by nipals

    """
    def __init__(self):
        pass

if __name__ == '__main__':
    a = [1, 2, 3, 4, 5, 6]
    arr = np.ndarray(shape=(2,3), buffer=np.array(a), dtype=int)
    print arr
    pca_case = PCABySVD(arr)
    pca_case.pca()
    print pca_case.loadings
    print pca_case.explained_var
    ev = pca_case.explained_var
    # reduce dimension acoording to the first N explained_var's sum.
    print ev[0]/sum(ev)
    # get the new arr in the mapping space.
    print np.dot(pca_case.loadings[0:,], arr)
