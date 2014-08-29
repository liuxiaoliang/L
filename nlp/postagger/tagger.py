#! /usr/bin/env python
# -*- encoding: utf-8 -*-
#
# Copyright (c) <2014> <leon.max.liew@gmail.com>
# 

"""pos tagger

A simple pos tagger using perceptron.
"""
__author__ = "xiaoliang liu"

import sys
sys.path.append("../../common")
sys.path.append("../../mllib")
import math
import random
import pickle
import numpy as np
import data_prepare as dpp
from classify.nn import *
from myexception import *


class PosTagger(object):
    """postagger

    """
    def __init__(self, cates=None, sample_file=None, result_file=None, model_file=None):
        """input file
        
        Args:
            cates: tagger classes
            sample_file: training or testing file.
            result_file: model file if training or predicting result if testing.
            model_file: only for predicting
            
        """
        self.sample_file = sample_file
        self.result_file = result_file
        self.model_file = model_file
        if cates:
            self.cates = cates
        else:
            self.cates = set()
        self.tagdict = {}
        self.model = Perceptron(self.cates)
        if sample_file:
            self.cdpp = dp.ConllDPP(self.sample_file)
        self.iter_num = 5

    def train(self):
        self._make_tagdict()
        self.model = Perceptron(self.cates)
        for i in range(self.iter_num):
            for words, tags in self.cdpp:
                self.train_one(words, tags)
            random.shuffle(self.cdpp)

    def predict(self):
        pass

    def train_one(self):
        prev, prev2 = self.cdpp.START
        context = self.cdpp.START + [self._normalize(w) for w in words] + self.cdpp.END
        for i, word in enumerate(words):
            guess = self.tagdict.get(word)
            if not guess:
                feats = self._get_feature(i, word, context, prev, prev2)
                guess = self.model.predict(feats)
                self.model.update(tags[i], guess, feats)
            prev2 = prev; prev = guess

    def predict_one(self, words, tokenize=True):
        prev, prev2 = START
        tags = DefaultList('') 
        context = START + [self._normalize(w) for w in words] + END
        for i, word in enumerate(words):
            tag = self.tagdict.get(word)
            if not tag:
                features = self._get_features(i, word, context, prev, prev2)
                tag = self.model.predict(features)
            tags.append(tag)
            prev2 = prev; prev = tag
        return tags
    
    def _get_feature(self):
        pass

    def _make_tagdict(self):
        counts = {}
        for s in self.cdpp:
            for word, tag in zip(sent[0], sent[1]):
                counts.setdefault(word, {})
                counts[word].setdefault(tag, 0)
                counts[word][tag] += 1
                self.cates.add(tag)
        freq_thresh = 20
        ambiguity_thresh = 0.97
        for word, tag_freqs in counts.items():
            tag, mode = max(tag_freqs.items(), key=lambda item: item[1])
            n = sum(tag_freqs.values())
            # Don't add rare words to the tag dictionary
            # Only add quite unambiguous words
            if n >= freq_thresh and (float(mode) / n) >= ambiguity_thresh:
                self.tagdict[word] = tag
                     
    def _normalize(self, word):
        if '-' in word and word[0] != '-':
            return '!HYPHEN'
        elif word.isdigit() and len(word) == 4:
            return '!YEAR'
        elif word[0].isdigit():
            return '!DIGITS'
        else:
            return word.lower()

    def save(self):
        pass

    def load(self):
        pass


if __name__ == '__main__':
    pass
