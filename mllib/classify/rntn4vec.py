#! /usr/bin/env python
# -*- encoding: utf-8 -*-
#
# Copyright (c) <2014> <leon.max.liew@gmail.com>
#

"""recursive neural tensor network

Implemention of algorithm descripted "Stanford Recursive Deep Models for 
sentiment compositionality over a Sentiment Treebank". Socker(2013).
"""
__author__ = "xiaoliang liu"

import sys
sys.path.append("../../common")
import math
import random
import numpy as np

class RNTN(object):
    """rnn model

    """
    
    def __init__(self, op=None, model=None, 
                 feature_file=None, dev_file=None, 
                 train_file=None, model_file=None):
        self.op = op # training options
        self.model = model # model parameters
        self.feature_file = feature_file
        self.dev_file = dev_file
        self.train_file = train_file
        self.model_file = model_file
        
        self.t = Timing()
        
    def develop(self):
        """adjust parameters

        """
        pass
    
    def train(self):
        """train wordvec and classified model

        """
        training_trees = 
        sum_grad_square = [] # init sum_grad_square
        batches_num = training_trees.size() / float(self.batch_size) + 1;
        
        # adagrad
        for epoch in range(self.epochs):
            print "starting epoch %d" % epoch
            if (epoch > 0 and self.adagrad_reset_frequency > 0 and 
                (epoch % self.adagrad_reset_frequency) == 0):
                sum_grad_square = [] # reset sum_grad_square
            shuffle(training_trees)
            for batch in range(batches_num):
                print "Epoch %d, batch %d" % (epoch, batch)
                tree_start = self.batch_size * batch
                tree_end = self.batch_size * (batch + 1)
                if tree_end + self.batch_size > train_trees.size():
                    tree_end = train_trees.size()
                self.one_batch_traning(self.rnnloss, training_trees[tree_start, tree_end], sum_grad_square)
                
                    
    
    def one_batch_training(self, rnnloss, rnntree, sum_grad_square):
        # adagrad
        for t in rnntree:
            

    def wordvec(self):
        """get distributed represent for sentence

        """
        pass

    def classify(self, cflie):
        """predict sentiment label

        """
        pass
    
    def softmax(self, w, nodevec):
        """softmax classifer
        
        Args:
            w: cate weight with size C * (N +1), including bias.
            nodevec: sample vector with size N + 1, including bias.
        
        Returns:
            A matrix with one column
        """
        resm = np.dot(w, nodevec)
        return resm/(resm.sum()+1)
    
    def get_cate(self, resm):
        """get cate
        
        Args:
            returned value from softmax

        Returns:
            index with max value in resm
        """
        argmax = 0
        for i in range(1, resm.size):
            if resm[i] > resm[argmax]:
                argmax = i
        return argmax

    def save_model(self):
        pass

    def load_model(self):
        pass


class RntnOptions(object):
    """training options

    """
    def __init__(self, batch_size=27, epochs=400, debug_output_epochs=8, 
                 max_train_time_seconds=60*60*24, learning_rate=0.1, 
                 adagrad_reset_frequency=1, regW=0.001, regV=0.001, 
                 regWs=0.0001, regL=0.0001):
        self.batch_size = batch_size # batch size in each training
        self.epochs = epochs # training epochs
        self.debug_output_epochs = debug_output_epochs
        self.max_train_time_seconds = max_train_time_seconds # 
        self.learning_rate = learning_rate
        self.adagrad_reset_frequency = adagrad_reset_frequency
        self.regW = regW # Regularization cost for the transform matrix
        self.regV = regV # Regularization cost for the transform tensor
        self.regWs = regWs # Regularization cost for the classification matrices
        self.regL = regL # Regularization cost for the word vectors

        
class RntnModel(object):
    """model parameters

    """
    def __init__(self, num_cate=5, num_hid=25):
        self.num_cate = num_cate # number of calsses
        self.num_hid = num_hid # Dimension of hidden layers, size of word vectors
        self.W = np.zeros(shape=(num_hid, 2*num_hid + 1), dtype=float) # transform
        self.V = np.zeros(shape=(2*num_hid, 2*num_hid, num_hid), dtype=float) # tensor
        self.Ws = np.zeros(shape=(num_cate, num_hid + 1), dtype=float) # cate weight
        self.L = {} # num_word * num_hid
    
    def randomW(self):
        # bias column values are initialized zero
        r = 1.0 / (math.sqrt(self.num_hid)*2.0)
        row, col = self.W.shape
        for i in range(row):
            for j in range(col-1):
                self.W[i, j] = random.randrange(-r, r)
    
    def randomV(self):
        r = 1.0 / (self.num_hid*4.0)
        row, col, s = self.V.shape
        for i in range(row):
            for j in range(col):
                for t in range(s):
                    self.V[i,j,t] = random.randrange(-r, r)
                                       
    def randomWs(self):
        # bias column values are initialized zero
        r = 1.0 / (math.sqrt(self.num_hid))
        row, col = self.W.shape
        for i in range(row):
            for j in range(col-1):
                self.Ws[i, j] = random.randrange(-r, r)
    
    def randomL(self):
        for w in self.L:
            for i in range(self.num_hid):
                self.L[w][i,0] = random.gauss(0,1)


class RntnLossAndGradient(object):
    """loss and gradient from rnn tree

    """
    def __init__(self, m, t):
        self.model = m
        self.sample = t
    
    


class SomeFunc(object):
    """common functions

    """
    def __init__(self):
        pass

    def tanh(self, x):
        """tanh
        
        Args:
            x: a matrix with float type

        Returns:
            tanh(x): also a matrix
        """
        row, col = x.shape
        out = np.ndarray(shape=(row, col),dtype=float)
        for i in range(row):
            for j in range(col):
                out[i, j] = math.tanh(x[i, j])
        return out


def rntn_train(dev_file, train_file, model_file):
    pass


def rntn_predict(test_file, model_file):
    pass


if __name__ == '__main__':
    pass
