#! /usr/bin/env python
# -*- encoding: utf-8 -*- 
#
# Copyright (c) <2014> <leon.max.liew@gmail.com>
#

"""svd

singular value decomposition.
"""

__author__ = 'xiaoliang liu'

import os
import sys
sys.path.append("../../common/")
import numpy as np

class SVD(object):
    """svd

    Svd form: A = U**Sigma**V.T, where U is a (m, r) matrix, S is
    a (r, r) matrix and V.T is a (r, n) matrix.
    """
    def __init__(self, arr):
        """

        Args:
            arr: numpy matrix with (m, n) dim.
        """
        self.arr = arr
        self.m, self.n = arr.shape
        
    def approximation(self, r):
        """
        
        Get a approximation svd of the matrix arr with the given rank.
        """
        u, s, vt = self.svd()
        u_prime = np.zeros((u.shape[0], r))
        s_prime = np.zeros((r, r))
        vt_prime = np.zeros((r, vt.shape[1]))
        for i in range(min(r, s.shape[0])):
            u_prime[..., i] = u[..., i]
            s_prime[i][i] = s[i][i]
            vt_prime[i,...] = vt[i,...]
        return (u_prime, s_prime, vt_prime)
    
    def svd(self):
        """svd

        Calculate the svd of input matrix.
        
        Returns:
            The tuple (U, S, V.T) form of SVD where U is a (m, r) matrix, S is
            a (r, r) matrix and V.T is a (r, n) matrix.
        """
        ata = np.dot(self.arr.T, self.arr) #A.T**A
        evals, evects = np.linalg.eigh(ata) # get eigenvals, eigenvectors of A.T**T
        svals, r = self.cal_svals(evals)
        s, v = self.decomp_matrix(svals, r, evects)
        sinv = np.linalg.inv(s)
        u = np.dot(self.arr, np.dot(v, sinv))
        return (u, s, v.T)
    
    def cal_svals(self, evals):
        """
        
        Convert eigenvalues into singular values. Store the original index for joining
        with corresponding eigenvectors.
        
        Returns:
            A data frame where D['singular_val'] is the singular value and D['index'] is 
            the index of the corresponding eigenvector of A.T**A.
        """
        svals = self.indexing_evals(evals)
        sorted_svals = self.sort_svals(svals)
        r = 0
        for s in np.nditer(sorted_svals, op_flags=['readwrite']):
            if s['sval'] > 0:
                s['sval'][...] = np.sqrt(s['sval'])
                r += 1
        return (sorted_svals, r)
    
    def indexing_evals(self, evals):
        """
        
        Save the index of eigenvalues to be used to join with corresponding eigenvalues.
        """
        dt = np.dtype([('sval', np.float64), ('index', np.uint16)])
        svals = np.zeros(evals.shape, dt) # singular values
        i = 0
        for e in np.nditer(evals):
            svals['sval'][i] = e
            svals['index'][i] = i
            i += 1
        return svals
        
    def sort_svals(self, svals):
        """
        
        Descending sort through a small thrick.
        """
        svals['sval'] *= -1
        sorted_svals = np.sort(svals, order='sval')
        sorted_svals['sval'] *= -1
        return sorted_svals
    
    def decomp_matrix(self, svals, r, evects):
        """
        
        Get S and V, where S is a (r, r) matrix and V is a (n, r) matrix.
        
        Args:
            svals: sigular values.
            r: number of no-zero singular values.
            evects: eigenvectors.
        Returns:
            A tuple (S, V).
        """
        s = np.zeros((r,r), dtype=np.float64)
        v = np.zeros((self.n, r), dtype=np.float64)
        for i in range(r):
            sv = svals['sval'][i]
            s[i][i] = sv
            ev = evects[:,svals['index'][i]]
            v[:,i] = ev
        return (s, v)
        
if __name__ == '__main__':
    a = [1, 2, 3, 4, 5, 6]
    arr = np.ndarray(shape=(2,3), buffer=np.array(a), dtype=int)
    print "------------matrix-------------"
    print arr
    # use numpy svd
    U, S, V = np.linalg.svd(arr, full_matrices=False)
    print "svd in numpy"
    print "-------------U-------------"
    print U
    print "-------------S-------------"
    print S
    print "-------------V-------------"
    print V
    # use svd above
    svd = SVD(arr)
    #U, S, V = svd.svd()
    U, S, V = svd.approximation(2)
    print "svd above"
    print "-------------U-------------"
    print U
    print "-------------S-------------"
    print S
    print "-------------V-------------"
    print V
