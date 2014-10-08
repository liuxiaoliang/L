#! /usr/bin/env python
# -*- encoding: utf-8 -*-
#
# Copyright (c) <2014> <leon.max.liew@gmail.com>
# 

"""kmeans

"""

__author__ = 'xiaoliang liu'

import sys
import math
sys.path.append("../../common/")
import data_prepare as dp

class Kmeans(objec):
    """kmeans

    A simple cluster method
    """
    def __init__(self, sample_file, k_num, output_filepath):
        self._filepath = sample_file
        self._k_num = k_num
        self._o_filepath = output_filepath
        self._dim = 0
        self._cdpp = dp.ClusterDPP(self._filepath)
        self._cdpp.load_sample()
        self._dim = self.cdpp.dim
        
    def _random_k(self):
        """random
        
        Generate random K points as the initial cluster center.
        """
        pass

    def _clusting(self):
        """clusting

        Assign each point to an index that corresponds to the index
        of the center point on it's proximity to that point. 
        """
        pass

    def _update(self):
        """update center

        """
        pass

    def _distance(self, a, b):
        """distance 
        
        Computer distance between a and b.
        """
        pass

    def _cmpt_cnt(self, points):
        """computer center

        """
        pass

    def kmeans(self):
        pass

    def save(self):
        pass
