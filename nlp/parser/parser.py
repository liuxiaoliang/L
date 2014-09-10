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

    Copy some codes from Github.
    """
    __SHIFT = 0; __RIGHT = 1; __LEFT = 2;
    __MOVES = (__SHIFT, __RIGHT, __LEFT)
    
    def __init__(self, sample_file=None, result_file=None, model_file=None,
                 tagger_model_file=None):
        """input file
        
        Args:
            sample_file: training or testing file.
            result_file: model file if training or predicting result if testing.
            model_file: only for predicting.
            tagger_model_file: postagger model
            
        """
        self.sample_file = sample_file
        self.result_file = result_file
        self.model_file = model_file
        if sample_file:
            self.cdpp = dpp.ConllDPP(self.sample_file)
            self.cdpp.get_data()
        self.model = Perceptron(self.__MOVES)
        self.iter_num = 15
        self.tagger = PosTagger(model_file=tagger_model_file)
        
    def train(self):
        self.tagger.load()
        for it in range(self.iter_num):
            corr = 0; total = 0
            random.shuffle(self.cdpp.data4parser)
            for words, gold_tags, gold_parse, gold_label in self.cdpp.data4parser:
                corr += self.train_one(it, words, gold_tags, gold_parse)
                total += len(words)
            #print it, '%.3f' % (float(corr) / float(total))
        self.model.average_weights()

    def train_one(self, it, words, gold_tags, gold_heads):
        n = len(words)
        i = 2
        stack = [1]
        deptree = self.DepTree(n)
        tags = self.tagger.predict_one(words)
        while stack or (i + 1) < n:
            features = self._get_feature(words, tags, i, n, stack, deptree)
            scores = self.model.score(features)
            valid_moves = self._get_valid_moves(i, n, len(stack))
            gold_moves = self._get_gold_moves(i, n, stack, deptree.heads, gold_heads)
            predict = max(valid_moves, key=lambda move: scores[move])
            assert gold_moves
            best = max(gold_moves, key=lambda move: scores[move])
            self.model.update(best, predict, features)
            i = self._transition(predict, i, stack, deptree)

    def predict(self):
        self.tagger.load()
        fp = open(self.result_file, 'w')
        for words, tags, gold_parse, gold_heads in self.cdpp.data4parser:
            predicted_heads = self.predict_one(words)
            for i, w in enumerate(words):
                fp.write(w + '\t' + gold_heads[i] + '\t' + predicted_heads[i] + '\n')
            fp.write('\n')
        fp.close()

    def predict_one(self, words):
        n = len(words)
        i = 2; stack = [1]; deptree = self.DepTree(n)
        tags = self.tagger.predict_one(words)
        while stack or (i+1) < n:
            features = self._get_feature(words, tags, i, n, stack, deptree)
            scores = self.model.score(features)
            valid_moves = self._get_valid_moves(i, n, len(stack))
            guess = max(valid_moves, key=lambda move: scores[move])
            i = self._transition(guess, i, stack, deptree)
        return deptree.heads
    
    class DepTree(object):
        """dependency tree
        
        Attributes:
            n: words lenght
            heads: node's head
            labels: nodes's dependency label
            lefts: left tree
            rights: right tree
            
        """
        def __init__(self, n):
            self.n = n
            self.heads = [None] * (n-1)
            self.labels = [None] * (n-1)
            self.lefts = []
            self.rights = []
            for i in range(n+1):
                self.lefts.append([])
                self.rights.append([])
 
        def add(self, head, child, label=None):
            self.heads[child] = head
            self.labels[child] = label
            if child < head:
                self.lefts[head].append(child)
            else:
                self.rights[head].append(child)
        
    def _get_feature(self, words, tags, n0, n, stack, deptree):
        """get feature

        Args:
            words: word list
            tags: tag list
            n0: buffer point
            n: words length
            stack: stack
            deptree: dependency tree
            
        Returns:
            features

        """
        def get_stack_context(depth, stack, data):
            if depth >= 3:
                return data[stack[-1]], data[stack[-2]], data[stack[-3]]
            elif depth >= 2:
                return data[stack[-1]], data[stack[-2]], ''
            elif depth == 1:
                return data[stack[-1]], '', ''
            else:
                return '', '', ''
     
        def get_buffer_context(i, n, data):
            if i + 1 >= n:
                return data[i], '', ''
            elif i + 2 >= n:
                return data[i], data[i + 1], ''
            else:
                return data[i], data[i + 1], data[i + 2]
     
        def get_parse_context(word, deps, data):
            if word == -1:
                return 0, '', ''
            deps = deps[word]
            valency = len(deps)
            if not valency:
                return 0, '', ''
            elif valency == 1:
                return 1, data[deps[-1]], ''
            else:
                return valency, data[deps[-1]], data[deps[-2]]
     
        features = {}
        # S0-2: Top three words on the stack
        # N0-2: First three words of the buffer
        # n0b1, n0b2: Two leftmost children of the first word of the buffer
        # s0b1, s0b2: Two leftmost children of the top word of the stack
        # s0f1, s0f2: Two rightmost children of the top word of the stack
        
        depth = len(stack)
        s0 = stack[-1] if depth else -1
        
        Wslist = get_stack_context(depth, stack, words) # Ws0, Ws1, Ws2
        Tslist = get_stack_context(depth, stack, tags) # Ts0, Ts1, Ts2
        
        Wnlist = get_buffer_context(n0, n, words) # Wn0, Wn1, Wn2
        Tnlist = get_buffer_context(n0, n, tags) # Tn0, Tn1, Tn2 
        
        Wnlplist = get_parse_context(n0, deptree.lefts, words) # Vn0b, Wn0b1, Wn0b2
        Tnlplist= get_parse_context(n0, deptree.lefts, tags) # Vn0b, Tn0b1, Tn0b2
        
        #Wnrplist = get_parse_context(n0, deptree.rights, words) # Vn0f, Wn0f1, Wn0f2
        #Tnrplist = get_parse_context(n0, deptree.rights, tags) # _, Tn0f1, Tn0f2
        
        Wslplist = get_parse_context(s0, deptree.lefts, words) # Vs0b, Ws0b1, Ws0b2
        Tslplist = get_parse_context(s0, deptree.lefts, tags) # _, Ts0b1, Ts0b2
        
        Wsrplist = get_parse_context(s0, deptree.rights, words) # Vs0f, Ws0f1, Ws0f2
        Tsrplist = get_parse_context(s0, deptree.rights, tags) # _, Ts0f1, Ts0f2
        
        # Cap numeric features at 5? 
        # String-distance
        Ds0n0 = min((n0 - s0, 5)) if s0 != 0 else 0
        
        features['bias'] = 1
        # Add word and tag unigrams
        for w in (Wnlist + Wslist + Wnlplist[1:] + Wslplist[1:] + Wsrplist[1:]):
            # (Wn0, Wn1, Wn2, Ws0, Ws1, Ws2, Wn0b1, Wn0b2, Ws0b1, Ws0b2, Ws0f1, Ws0f2)
            if w:
                features['w=%s' % w] = 1
        for t in (Tnlist + Tslist + Tnlplist[1:] + Tslplist[1:] + Tsrplist[1:]):
            # (Tn0, Tn1, Tn2, Ts0, Ts1, Ts2, Tn0b1, Tn0b2, Ts0b1, Ts0b2, Ts0f1, Ts0f2)
            if t:
                features['t=%s' % t] = 1
        
        # Add word/tag pairs
        for i, (w, t) in enumerate(zip(Wnlist + Wslist[:1], Tnlist + Tslist[:1])):
            # ((Wn0, Tn0), (Wn1, Tn1), (Wn2, Tn2), (Ws0, Ts0))
            if w or t:
                features['%d w=%s, t=%s' % (i, w, t)] = 1
        
        # Add some bigrams
        features['s0w=%s,  n0w=%s' % (Wslist[0], Wnlist[0])] = 1 # Ws0, Wn0
        features['wn0tn0-ws0 %s/%s %s' % (Wnlist[0], Tnlist[0], Wslist[0])] = 1 # Wn0, Tn0, Ws0
        features['wn0tn0-ts0 %s/%s %s' % (Wnlist[0], Tnlist[0], Tslist[0])] = 1 # Wn0, Tn0, Ts0
        features['ws0ts0-wn0 %s/%s %s' % (Wslist[0], Tslist[0], Wnlist[0])] = 1 # Ws0, Ts0, Wn0
        features['ws0-ts0 tn0 %s/%s %s' % (Wslist[0], Tslist[0], Tnlist[0])] = 1 # Ws0, Ts0, Tn0
        features['wt-wt %s/%s %s/%s' % (Wslist[0], Tslist[0], Wnlist[0], Tnlist[0])] = 1 # Ws0, Ts0, Wn0, Tn0
        features['tt s0=%s n0=%s' % (Tslist[0], Tnlist[0])] = 1 # Ts0, Tn0
        features['tt n0=%s n1=%s' % (Tnlist[0], Tnlist[1])] = 1 # Tn0, Tn1
        
        # Add some tag trigrams
        trigrams = (tuple(Tnlist), (Tslist[0], Tnlist[0], Tnlist[1]), (Tslist[0], Tslist[1], Tnlist[0]), 
                    (Tslist[0], Tsrplist[1], Tnlist[0]), (Tslist[0], Tsrplist[1], Tnlist[0]), 
                    (Tslist[0], Tnlist[0], Tnlplist[1]),(Tslist[0], Tslplist[1], Tslplist[2]), 
                    (Tslist[0], Tsrplist[1], Tsrplist[2]), (Tnlist[0], Tnlplist[1], Tnlplist[2]),
                    tuple(Tslist[0]))
        for i, (t1, t2, t3) in enumerate(trigrams):
            if t1 or t2 or t3:
                features['ttt-%d %s %s %s' % (i, t1, t2, t3)] = 1
        
        # Add some valency and distance features
        vw = ((Wslist[0], Wsrplist[0]), (Wslist[0], Wslplist[0]), (Wnlist[0], Wnlplist[0]))
        vt = ((Tslist[0], Wsrplist[0]), (Tslist[0], Wslplist[0]), (Tnlist[0], Wnlplist[0]))
        d = ((Wslist[0], Ds0n0), (Wnlist[0], Ds0n0), (Tslist[0], Ds0n0), (Tnlist[0], Ds0n0),
             ('t' + Tnlist[0]+Tslist[0], Ds0n0), ('w' + Wnlist[0]+Wslist[0], Ds0n0))
        for i, (w_t, v_d) in enumerate(vw + vt + d):
            if w_t or v_d:
                features['val/d-%d %s %d' % (i, w_t, v_d)] = 1
        return features
    
    def _get_valid_moves(self, i, n, sn):
        """get valid moves

        Args:
            i: buffer point
            n: word lenght
            sn: stack_depth
            
        Returns:
            moves: valid moves 
        """
        moves = []
        if (i+1) < n:
            moves.append(self.__SHIFT)
        if sn >= 2:
            moves.append(self.__RIGHT)
        if sn >= 1:
            moves.append(self.__LEFT)
        return moves
    
    def _get_gold_moves(self, i, n, stack, dtheads, gheads):
        """get gold moves
        
        Args:
            i: buffer point
            n: word length
            stack:
            dtheads: deptree's heads
            gheads: real heads
        Returns:
            moves: gold moves
        """
        def deps_between(target, others, gold):
            for word in others:
                if gold[word] == target or gold[target] == word:
                    return True
            return False
 
        valid = self._get_valid_moves(i, n, len(stack))
        if not stack or (self.__SHIFT in valid and gheads[i] == stack[-1]):
            return [self.__SHIFT]
        if gheads[stack[-1]] == i:
            return [self.__LEFT]
        costly = set([m for m in self.__MOVES if m not in valid])
        # If the word behind s0 is its gold head, Left is incorrect
        if len(stack) >= 2 and gheads[stack[-1]] == stack[-2]:
            costly.add(self.__LEFT)
        # If there are any dependencies between i and the stack,
        # pushing n0 will lose them.
        if self.__SHIFT not in costly and deps_between(i, stack, gheads):
            costly.add(self.__SHIFT)
        # If there are any dependencies between s0 and the buffer, popping
        # s0 will lose them.
        if deps_between(stack[-1], range(i+1, n-1), gheads):
            costly.add(self.__LEFT)
            costly.add(self.__RIGHT)
        return [m for m in self.__MOVES if m not in costly]

    def _transition(self, predict, i, stack, deptree):
        """transition
        
        Args:
            predict: predicted transition
            i: buffer point
            stack
            deptree: dependency tree
            
        """
        if predict == self.__SHIFT:
            stack.append(i)
            return i + 1
        elif predict == self.__RIGHT:
            deptree.add(stack[-2], stack.pop())
            return i
        elif predict == self.__LEFT:
            deptree.add(i, stack.pop())
            return i
        assert predict in self.__MOVES

    def save(self):
        with open(self.result_file, 'wb') as fp:
            pickle.dump(self.model.weights, fp)

    def load(self):
        with open(self.model_file, 'rb') as fp:
            m = pickle.load(fp)
            self.model.weights = m


if __name__ == '__main__':
    sample_file = sys.argv[1]
    result_file = sys.argv[2]
    model_file = sys.argv[3]
    tagger_model_file = sys.argv[4]
    # train
    # use ../../dataset/conll2007/eus.train
    pt = Parser(sample_file=sample_file, result_file=result_file, tagger_model_file=tagger_model_file)
    pt.train()
    pt.save()
    # predict
    # use ../../dataset/conll2007/eus.test
    #pt = Parser(sample_file=sample_file, result_file=result_file, model_file=model_file, tagger_model_file=tagger_model_file)
    #pt.load()
    #pt.predict()
