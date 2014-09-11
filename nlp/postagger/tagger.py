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
        self.cdpp = dpp.ConllDPP(self.sample_file)
        self.iter_num = 5

    def train(self):
        self.cdpp.get_data()
        self._make_tagdict()
        self.model = Perceptron(self.cates)
        for i in range(self.iter_num):
            for words, tags in self.cdpp.data4postagger:
                self.train_one(words, tags)
            random.shuffle(self.cdpp.data4postagger)
        self.model.average_weights()
        
    def predict(self):
        self.cdpp.get_data()
        fp = open(self.result_file, 'w')
        for words, tags in self.cdpp.data4postagger:
            predicted_tags = self.predict_one(words)
            for i, w in enumerate(words):
                fp.write(w + '\t' + tags[i] + '\t' + predicted_tags[i] + '\n')
            fp.write('\n')
        fp.close()
        
    def train_one(self, words, tags):
        prev, prev2 = self.cdpp.START
        context = self.cdpp.START + [self.cdpp.normalize(w) for w in words] + self.cdpp.END
        for i, word in enumerate(words):
            guess = self.tagdict.get(word)
            if not guess:
                feats = self._get_feature(i, word, context, prev, prev2)
                guess = self.model.predict(feats)
                self.model.update(tags[i], guess, feats)
            prev2 = prev; prev = guess

    def predict_one(self, words, tokenize=True):
        prev, prev2 = self.cdpp.START
        tags = []
        context = self.cdpp.START + [self.cdpp.normalize(w) for w in words] + self.cdpp.END
        for i, word in enumerate(words):
            tag = self.tagdict.get(word)
            if not tag:
                features = self._get_feature(i, word, context, prev, prev2)
                tag = self.model.predict(features)
            tags.append(tag)
            prev2 = prev; prev = tag
        return tags
    
    def _get_feature(self, i, word, context, prev, prev2):
        """get featufe
        
        Map tokens into a feature representation, implemented as a
        {hashable: float} dict. If the features change, a new model must be
        trained.
        
        This function is copied from Github.
        """
        def add(name, *args):
            feas.setdefault(' '.join((name,) + tuple(args)), 0)
            feas[' '.join((name,) + tuple(args))] += 1
 
        i += len(self.cdpp.START)
        feas = {}

        add('bias')
        add('i suffix', word[-3:])
        add('i pref1', word[0])
        add('i-1 tag', prev)
        add('i-2 tag', prev2)
        add('i tag+i-2 tag', prev, prev2)
        add('i word', context[i])
        add('i-1 tag+i word', prev, context[i])
        add('i-1 word', context[i-1])
        add('i-1 suffix', context[i-1][-3:])
        add('i-2 word', context[i-2])
        add('i+1 word', context[i+1])
        add('i+1 suffix', context[i+1][-3:])
        add('i+2 word', context[i+2])
        return feas

    def _make_tagdict(self):
        counts = {}
        for s in self.cdpp.data4postagger:
            for word, tag in zip(s[0], s[1]):
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
                     
    def save(self):
        with open(self.result_file, 'wb') as fp:
            pickle.dump((self.model.weights, self.tagdict, self.cates), fp)

    def load(self):
        with open(self.model_file, 'rb') as fp:
            m = pickle.load(fp)

        self.model.weights = m[0]
        self.tagdict = m[1]
        self.cates = m[2]
        self.model.cates = m[2]


if __name__ == '__main__':
    sample_file = sys.argv[1]
    result_file = sys.argv[2]
    model_file = sys.argv[3]
    # train
    # use ../../dataset/conll2007/eus.train
    #pt = PosTagger(sample_file=sample_file, result_file=result_file)
    #pt.train()
    #pt.save()
    # predict
    # use ../../dataset/conll2007/eus.test
    pt = PosTagger(sample_file=sample_file, result_file=result_file, model_file=model_file)
    pt.load()
    pt.predict()
