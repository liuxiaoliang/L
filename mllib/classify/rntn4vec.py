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
import numpy as np

class RNTN(object):
    """rnn model

    """
    
    def __init__(self,num_cate=0,num_hid=0,max_train_time_seconds=60*60*24,
                 batch_size=27,epochs=400,adagrad_reset_frequency=1,
                 dev_file=None,train_file=None):
        # parameters options
        self.num_cate = num_cate # number of calsses
        self.num_hid = num_hid # Dimension of hidden layers, size of word vectors
        self.wordvec = {}
        self.rnnloss = RNNLoss() # include V and W in activation function.
        self.cate_weight = {}
        # training options
        self.max_train_time_milliseconds = max_train_time_seconds * 1000
        self.batch_size = batch_size
        self.epochs = epochs # Number of times through all the trees
        self.adagrad_reset_frequency = adagrad_reset_frequency
        # other
        self.t = Timing()
        
    def develop(self):
        """adjust parameters

        """
        
        pass
    
    def train(self):
        """train wordvec and classified model

        """
        training_trees = None
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


class RNNLoss(object):
    """loss from rnn tree

    """
    pass


class ActFunc(object):
    """activation function

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

