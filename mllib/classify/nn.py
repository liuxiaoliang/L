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
        # a list storing label set
        self.cates = cates
        # model file path
        self.path = path
        self.weights = {}
        # The accumulated values, for the averaging. 
        self._total = {}
        # The last time the feature was changed.
        self._last_time = {}
        # Number of samples
        self.i = 0
        
        
    def update(self, real, predict, fea):
        def update_fea(f, c, w, v):
            param = (f, c)
            self._total[param].setdefault(param, 0)
            self._total[param] += (self.i - self._last_time[param]) * w
            self._last_time[param] = self.i
            self.weights[f][c] = w + v
 
        self.i += 1
        if real == predict:
            return None
        for f in fea:
            weights = self.weights.setdefault(f, {})
            update_fea(f, real, weights.get(truth, 0.0), 1.0)
            update_fea(f, predict, weights.get(guess, 0.0), -1.0)
    
    def average_weights(self):
        if self.i == 0:
            return None
        for fea, weights in self.weights.items():
            new_fea_weights = {}
            for c, w in weights.items():
                param = (fea, c)
                total = self._total[param]
                total += (self.i - self._last_time[param]) * weight
                averaged_weights = round(total / float(self.i), 3)
                if averaged:
                    new_fea_weights[c] = averaged_weights
            self.weights[fea] = new_fea_weights

    def predict(self, fea):
        scores = dict((c, 0) for c in self.cates)
        for f, v in fea.items():
            if v == 0:
                continue
            if f not in self.weights:
                continue
            fea_weights = self.weights[f]
            for c, w in fea_weights.items():
                scores[c] += v * w
        return max(self.cates, key=lambda c: (scores[c], c))

    def save(self):
        with open(path, 'w') as fp:
            pickle.dump(self.weights, fp)

    def load(self):
        with open(path, 'w') as fp:
            self.weights = pickle.load(fp)


if __name__ == '__main__':
    # test
    pass
