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

class RNTN(object):
    """rnn model

    """
    
    def __init__(self,num_cate=None,num_hid=None,max_train_time_seconds=60*60*24,
                 batch_size=27,epochs=400,adagrad_reset_frequency=1,
                 dev_file=None,train_file=None):
        # parameters options
        self.num_cate = num_cate # number of calsses
        self.num_hid = num_hid # Dimension of hidden layers, size of word vectors
        
        # training options
        self.max_train_time_milliseconds = max_train_time_seconds * 1000
        self.batch_size = batch_size
        self.epochs = epochs # Number of times through all the trees
        self.adagrad_reset_frequency = adagrad_reset_frequency
        
        self.rnnloss = RNNLoss()
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
                sum_grad_square = self.one_batch_traning(self.rnnloss, training_trees[tree_start, tree_end], sum_grad_square)
                
                    
    def one_batch_training(self, rnnloss, rnntree, sum_grad_square):
        # adagrad
        pass

    def wordvec(self):
        """get distributed represent for sentence

        """
        pass

    def classify(self, cflie):
        """predict sentiment label

        """
        pass
    
    def save_model(self):
        pass

    def load_model(self):
        pass


class RNNLoss(object):
    """loss from rnn tree

    """
    pass


class ActFunc(object):
    """activate function

    """
    pass


class LossFunc(object):
    """loss function

    """
    def __init__(self):
        pass
