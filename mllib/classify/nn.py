#! /usr/bin/env python
# -*- encoding: utf-8 -*-
#
# Copyright (c) <2014> <leon.max.liew@gmail.com>
# 

"""neural network

Because nn is a online algorithm, it's training procedure is going in
some specific application unlike bayes or svm and so on.

Algo:
1) Averaged Perceptron

"""
__author__ = "xiaoliang liu"

import sys
sys.path.append("../../common")
import math
import random
import pickle
import numpy as np
import data_prepare as dpp
from myexception import *

class Perceptron(object):
    """perceptron
    
    """
    def __init__(self, cates=None, path=None):
        self.cates = cates
        self.path = path
        self.weights = {}
        # The accumulated values, for the averaging. 
        self._total = {}
        # The last time the feature was changed.
        self._last_time = {}
        # Number of samples
        self.i = 0
        
        
    def update(self, real, predict, fea):
        pass
    
    def average_weights(self):
        pass

    def predict(self, fea):
        pass

    def save(self):
        pass

    def load(self):
        pass
