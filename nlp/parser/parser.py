#! /usr/bin/env python
# -*- encoding: utf-8 -*-
#
# Copyright (c) <2014> <leon.max.liew@gmail.com>
# 

"""parser

A parser descripted in paper "Training Deterministic Parsers with
Non-Deterministic Oracles", using arc-hybrid transition system.

Arc-hybrid system is descripted in "Dynamic programming algorithms 
for transition-based dependency parsers."
"""
__author__ = "xiaoliang liu"

import sys
sys.path.append("../../common")
sys.path.append("../../mllib")
sys.path.append("../postagger/")
import math
import random
import pickle
import numpy as np
import data_prepare as dpp
from classify.nn import *
from myexception import *
from tagger import *

class Parser(object):
    """parser

    """
    __SHIFT = 0; __RIGHT = 1; __LEFT = 2;
    __MOVES = (__SHIFT, __RIGHT, __LEFT)
    def __init__(self, sample_file=None, result_file=None, model_file=None
                 tagger_model_file=None):
        """input file
        
        Args:
            sample_file: training or testing file.
            result_file: model file if training or predicting result if testing.
            model_file: only for predicting.
        """
        self.sample_file = sample_file
        self.result_file = sample_file
        self.model_file = model_file
        if sample_file:
            self.cdpp = dp.ConllDPP(self.sample_file)
        self.model = Perceptron(__MOVES)
        self.iter_num = 15
        self.tagger = PosTagger(model=tagger_model_file)
        
    def train(self):
        self.tagger.load()
        for it in range(self.iter_num):
            corr = 0; total = 0
            random.shuffle(self.cdpp)
            for words, gold_tags, gold_parse, gold_label in self.cdpp:
                corr += self.train_one(itn, words, gold_tags, gold_parse)
                total += len(words)
            print it, '%.3f' % (float(corr) / float(total))
        self.model.average_weights()

    def train_one(self):
        

    def predict(self):
        pass

    def predict_one(self):
        pass
    
    def _get_feature(self):
        pass
    
    def save(self):
        pass

    def load(self):
        pass


if __name__ == '__main__':
    pass
