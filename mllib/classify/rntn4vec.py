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
import data_prepare as dpp

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
        self.t = Timing() # Timing class
        self.trd = None # train or test dpp handle
        self.drd = None # dev dpp handle

    def load_data(self):
        """Load data

        """
        # read trees
        if self.train_file:
            self.trd = dpp.RNNDPP(self.feature_file, self.train_file)
            self.trd.load_feature()
            self.trd.load_sample()
        if self.dev_file:
            self.drd = dpp.RNNDPP(self.feature_file, self.dev_file)
            self.drd.load_feature()
            self.drd.load_sample()
        
    def train(self):
        """train wordvec and classified model

        """
        self.t.start()
        sum_grad_square = [0.0] * self.model.total_param_size() # init sum_grad_square
        if self.op.batch_size == 0:
            self.op.batch_size = 1
        batches_num = int(len(self.trd.slist) / float(self.op.batch_size));
        # adagrad
        for epoch in range(self.op.epochs):
            print "starting epoch %d" % epoch
            if (epoch > 0 and self.op.adagrad_reset_frequency > 0 and 
                (epoch % self.op.adagrad_reset_frequency) == 0):
                sum_grad_square = [0.0] * self.model.total_param_size() # reset sum_grad_square
            shuffle_tree_list = self.trd.slist
            random.shuffle(shuffle_tree_list)
            for batch in range(batches_num):
                print "Epoch %d, batch %d" % (epoch, batch)
                tree_start = self.op.batch_size * batch
                tree_end = self.op.batch_size * (batch + 1)
                if tree_end + self.op.batch_size > len(shuffle_tree_list):
                    tree_end = len(shuffle_tree_list)
                self.one_batch_traning(shuffle_tree_list[tree_start:tree_end], sum_grad_square)
                # computer timing
                elapsed_time = self.t.report()
                print "epoch is %d, batch is %d, elapsed time %d" % (epoch, batch, elapsed_time)
                if(self.op.max_train_time_seconds > 0 and elapsed_time > self.op.max_train_time_seconds*1000):
                    break
                if (batch==0 and epoch > 0 and epoch % self.op.debug_output_epochs == 0):
                    score = 0.0
                    pass
            elapsed_time = self.t.report()
            if(self.op.max_train_time_seconds > 0 and elapsed_time > self.op.max_train_time_seconds*1000):
                print "Max training time exceeded, exiting"
                break
            
    def one_batch_training(self, t, sum_grad_square):
        # adagrad
        rlg = RntnLossAndGradient(self.model, t, self.op)
        theta = self.model.param2vector()
        eps = 1e-3
        cur_loss = 0
        gradf = rlg.derivative_at(theta)
        cur_loss = rlg.value_at(theta)
        print "cur batch loss %f" % cur_loss
        for f in range(len(gradf)):
            sum_grad_square[f] = sum_grad_square[f] + gradf[f]*gradf[f]
            theta[f] = theta[f] - (self.op.learning_rate * gradf[f]/(math.sqrt(sum_grad_square[f])+eps))
        self.model.vector2param(theta)    
        
    def wordvec(self):
        """get distributed represent for sentence

        """
        pass

    def classify(self, cflie):
        """predict sentiment label

        """
        pass
       
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
                 regWs=0.0001, regL=0.0001, isEval=True):
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
    def __init__(self, num_cate=5, num_hid=25, feature_file=None):
        self.num_cate = num_cate # number of calsses
        self.num_hid = num_hid # Dimension of hidden layers, size of word vectors
        self.W = np.zeros(shape=(num_hid, 2*num_hid + 1), dtype=float) # transform
        self.V = np.zeros(shape=(2*num_hid, 2*num_hid, num_hid), dtype=float) # tensor
        self.Ws = np.zeros(shape=(num_cate, num_hid + 1), dtype=float) # cate weight
        self.L = {} # num_word * num_hid
        self.wlist = [] # word list
        rd = dpp.RNNDPP(feature_file, '')
        rd.load_feature()
        # init L in model
        for fid in rd.id2feature:
            self.wlist.append(fid)
            self.model.L[fid] = np.zeros(shape=(num_hid, 1), dtype=float)
            
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
        row, col = self.Ws.shape
        for i in range(row):
            for j in range(col-1):
                self.Ws[i, j] = random.randrange(-r, r)
    
    def randomL(self):
        for w in self.L:
            for i in range(self.num_hid):
                self.L[w][i,0] = random.gauss(0,1)
    
    def total_param_size(self):
        return self.W.size + self.V.size + self.Ws.size + len(self.L) * self.num_hid

    def param2vector(self):
        total_size = self.total_param_size()
        theta = [0.0] * total_size
        i = 0
        try:
            for t in self.W.flat:
                theta[i] = t
                i += 1
            for t in self.V.flat:
                theta[i] = t
                i += 1
            for t in self.Ws.flat:
                theta[i] = t
                i += 1
            for w in self.wlist:
                for t in self.L[w].flat:
                    theta[i] = t
                    i += 1
        except IndexError, e:
            sys.stderr.write("param2vector error, total_param_size is %d, current theta index is %d, error is %s \n" 
                             % (total_size, i, str(e)))
            exit(1)
        if i != total_size:
            sys.stderr.write("param2vector error, total_param_size is %d, current theta index is %d \n",
                             % (total_size, i))
            exit(1)
    
    def vector2param(theta):
        total_size = self.total_param_size()
        if len(theta) != total_size:
            sys.stderr.write("vector2param error, len(theta) is %d, total_param_size is %d \n",
                             %(len(theta), total_size))
            exit(1)
         index = 0
         # get W
         row, col = self.W.shape
         self.W = np.ndarray(shape=(row, col), buffer=array(theta[index:index+row*col]), dtype=float)
         index += row*col
         # get V
         row, col, s = self.V.shape
         self.V = np.ndarray(shape=(row,col,s), buffer=array(theta[index:index+row*col*s]), dtype=float)
         index += row*col*s
         # get Ws
         row, col = self.Ws.shape
         self.Ws = np.ndarray(shape(row,col), buffer=array(theta[index:index+row*col]), dtype=float)
         index += row*col
         # get L
         row = self.num_hid
         for w in self.wlist:
             self.L[w] = nd.ndarray(shape(row, 1), buffer=array(theta[index:index+row]), dtype=float)
             index += row
         
         
class RntnLossAndGradient(object):
    """loss and gradient from rnn tree

    """
    def __init__(self, m, t, op):
        self.model = m
        self.sample = t
        self.op = op
        self.derivative = [] # derivative
        self.value = 0.0 # value
        self.f = SomeFunc()
        
    def derivative_at(self, theta):
        self.calculate(theta)
        return derivative
    
    def value_at(self, theta):
        self.calculate(theta)
        return value

    def calculate(self, theta):
        self.model.vector2param(theta)
        for t in self.sample:
            forward_propagate(t, t.root)
            
    
    def forward_propagate(self, tree, cur_point):
        nodevector = None
        cate = None
        if cur_point is None:
            exit(1)
        if tree.nodelist[cur_point].isLeaf():
            cate = tree.nodelist[cur_point].nlabel
            if tree.nodelist[cur_point].fid:
                wordvector = self.model.L[fid]
                nodevector = self.f.tanh(wordvector)
        elif(tree.nodelist[cur_point].left and tree.nodelist[cur_point].right):
            self.forward_propagate(tree,tree.nodelist[cur_point].left)
            self.forward_propagate(tree,tree.nodelist[cur_point].right)
            leftvector = tree.nodelist[tree.nodelist[cur_point].left].nodevec
            rightvector = tree.nodelist[tree.nodelist[cur_point].right].nodevec
            childrenvector = self.f.concatenate_with_bias(leftvector, rightvector)
            tensor_in = self.f.concatenate(leftvector, rightvector)
            tensor_out = self.f.bilinear_products(self.model.V, tensor_in)
            nodevector = self.f.tanh(np.dot(self.W,childrenvector) + tensor_out)
        elif(tree.nodelist[cur_point].left):
            self.forward_propagate(tree,tree.nodelist[cur_point].left)
            leftvector = tree.nodelist[tree.nodelist[cur_point].left].nodevec
            nodevector = self.f.tanh(leftvector)
        #elif(tree.nodelist[cur_point].right):
        # tree is created by preorder traversal, so never gone here
        predictions = self.f.softmax(np.dot(self.model.Ws, self.f.concatenate_with_bias(nodevector)))
        tree.nodelist[cur_point].nodevector = nodevector
        tree.nodelist[cur_point].prediction = prediction
        tree.nodelist[cur_point].index = self.f.get_predicted_cate(prediction)


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

    def concatenate_with_bias(self, lvec, rvec):
        # bias is 1.0
        new_array_row = lvec.shape[0] + rvec.shape[0]
        new_array = np.zeros(shope=(new_array_row, 1), dtype=float)
        index = 0
        for i in range(lvec.shape[0]):
            new_array[index][0] = lvec[i][0]
            index += 1
        for i in range(rvec.shape[0]):
            new_array[index][0] = rvec[i][0]
            index += 1
        new_array[index][0] = 1.0 # bias
    
        return new_array

    def concatenate(self, lvec, rvec):
        new_array_row = lvec.shape[0] + rvec.shape[0]
        new_array = np.zeros(shope=(new_array_row, 1), dtype=float)
        index = 0
        for i in range(lvec.shape[0]):
            new_array[index][0] = lvec[i][0]
            index += 1
        for i in range(rvec.shape[0]):
            new_array[index][0] = rvec[i][0]
            index += 1
        
        return new_array
    
    def bilinear_products(self, t, vec):
        row, col, s = t.shape
        out = np.zeros(shape=(s, 1), dtype=float)
        for i in range(s):
            out[i][0] = np.dot(np.dot(vec.transpose(), t[:,:,i]), vec)
        return out
    
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

    def get_predicted_cate(self, vec):
        index = 0
        for i in range(vec.size):
            if vec[i][0] > index:
                index = i
        return index


def rntn_train(feature_file, dev_file, train_file, model_file):
    pass


def rntn_predict(test_file, model_file):
    pass


if __name__ == '__main__':
    pass
