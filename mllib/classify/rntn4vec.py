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
import pickle
import numpy as np
import data_prepare as dpp
from myexception import *

class RNTN(object):
    """rnn model

    """

    def __init__(self, op=None, model=None,
                 feature_file=None, dev_file=None,
                 train_file=None, model_file=None,
                 result_file=None):
        self.op = op # training options
        self.model = model # model parameters
        self.feature_file = feature_file
        self.dev_file = dev_file
        self.train_file = train_file # train or test file
        self.model_file = model_file
        self.t = Timing() # Timing class
        self.trd = None # train or test dpp handle
        self.drd = None # dev dpp handle
        self.result_file = result_file # predicted result

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
        print self.t.report()

    def one_batch_training(self, t, sum_grad_square):
        # adagrad
        rlg = RntnLossAndGradient(self.op, self.model, t)
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

    def predict(self):
        """predict sentiment label

        """
        self.t.start()
        rlg = RntnLossAndGradient(self.op, self.model, None)
        fp = open(self.result_file, 'w')
        for t in self.trd.slist:
            rlg.forward_propagate(t, t.root)
            r = t.get_predicted_label()
            for ri in r:
                fp.write("%s,%s\n" % (ri[0], ri[1]))
        fp.close()
        print self.t.report()

    def save_model(self):
        # serialization
        fp = open(self.model_file, 'w')
        pickle.dump(self.model, fp)
        fp.close()

    def load_model(self):
        # deserialization
        fp = open(self.model_file, 'r')
        self.model = pickle.load(fp)
        fp.close()


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
         self.Ws = np.ndarray(shape=(row,col), buffer=array(theta[index:index+row*col]), dtype=float)
         index += row*col
         # get L
         row = self.num_hid
         for w in self.wlist:
             self.L[w] = nd.ndarray(shape=(row, 1), buffer=array(theta[index:index+row]), dtype=float)
             index += row


class RntnLossAndGradient(object):
    """loss and gradient from rnn tree

    """
    def __init__(self, op, m, t):
        self.model = m
        self.sample = t
        self.op = op
        self.derivative = [] # derivative
        self.value = 0.0 # value
        self.f = SomeFunc()
        # derivative
        self.W_d = np.zeros(shape=(self.model.num_hid, self.model.num_hid*2+1), dtype=float)
        self.V_d = np.zeros(shape=(self.model.num_hid*2, self.model.num_hid*2, self.model.num_hid), dtype=float)
        self.Ws_d = np.zeros(shape=(self.model.num_cate, self.model.num_hid+1), dtype=float)
        self.wordvec_d = {}
        for w in self.model.L:
            self.wordvec_d[w] = np.zeros(shape=(self.model.num_hid, 1))

    def derivative_at(self, theta):
        self.calculate(theta)
        return derivative

    def value_at(self, theta):
        self.calculate(theta)
        return value

    def calculate(self, theta):
        self.model.vector2param(theta)
        forword_tree = []
        for t in self.sample:
            forward_propagate(t, t.root)
            forword_tree.append(t)
                                         
        error = 0.0
        for t in forword_tree:
            delta = np.zeros(shape=(self.model.num_hid, 1), dtype=float)
            backprop_derivatives_and_error(t, t.root, delta)
            error += t.get_error_sum()
        scale = 1.0 / self.op.batch_size
        self.value = scale * error
        
        self.value += self.scale_and_regularize(self.W_d, self.model.W, scale, self.op.regW)
        self.value += self.scale_and_regularize(self.V_d, self.model.V, scale, self.op.regV)
        self.value += self.scale_and_regularize(self.Ws_d, self.model.Ws, scale, self.op.regWs)
        self.value += self.scale_and_regularize4wordvec(self.wordvec_d, self.model.L, scale, self.op.regL)
        
        self.derivative = self.f.params2vector(theta.size, self.W_d, self.V_d, self.Ws_d, self.wordvec_d, self.model)
        
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
            nodevector = self.f.tanh(np.dot(self.W, childrenvector) + tensor_out)
        elif(tree.nodelist[cur_point].left):
            self.forward_propagate(tree,tree.nodelist[cur_point].left)
            leftvector = tree.nodelist[tree.nodelist[cur_point].left].nodevec
            nodevector = self.f.tanh(leftvector)
        #elif(tree.nodelist[cur_point].right):
        # tree is created by preorder traversal, so never gone here
        empty_array = np.empty(shape=(0,0))
        prediction = self.f.softmax(np.dot(self.model.Ws, self.f.concatenate_with_bias(nodevector, empty_array)))
        tree.nodelist[cur_point].nodevector = nodevector
        tree.nodelist[cur_point].prediction = prediction
        tree.nodelist[cur_point].index = self.f.get_predicted_cate(prediction)
        
    def backprop_derivatives_and_error(self, tree, cur_point, delta):
        currentvector = tree.nodelist[cur_point].nodevector
        gold_label = np.zeros(shape=(self.model.num_cate, 1), dtype=float)
        cindex = tree.nodelist[cur_point].index
        if cindex >= 0:
            gold_label[cindex][0] = 1.0
        prediction = tree.nodelist[cur_point].prediction
        delta_class = prediction - gold_label
        empty_array = np.empty(shape=(0,0))
        local_class_delta = np.dot(delta_class, self.f.concatenate_with_bias(currentvector, empty_array).transpose())
        error = -self.f.element_mult(self.f.elment_wise_log(prediction), gold_label).sum()
        tree.nodelist[cur_point].error = error
        
        if tree.nodelist[cur_point].isLeaf():
            self.Ws_d += local_class_delta
            fid = tree.nodelist[cur_point].fid
            if fid:
               currentvector_derivative = self.f.element_wise_tanh_derivative(currentvector)
               delta_from_class = np.dot(self.model.Ws.transpose(), delta_class)
               delta_from_class = self.f.element_mult(delta_from_class[0:self.model.num_hid, 0:1], currentvector_derivative)
               delta_full = delta_from_class + delta
               self.wordvec_d[fid] += delta_full
        elif(tree.nodelist[cur_point].left and tree.nodelist[cur_point].right):
            self.Ws_d += local_class_delta
            currentvector_derivative = self.f.element_wise_tanh_derivative(currentvector)
            delta_from_class = np.dot(self.model.Ws.transpose(), delta_class)
            delta_from_class = self.f.element_mult(delta_from_class[0:self.model.num_hid, 0:1], currentvector_derivative)
            delta_full = delta_from_class + delta
            leftvector = tree.nodelist[tree.nodelist[cur_point].left].nodevec
            rightvector = tree.nodelist[tree.nodelist[cur_point].right].nodevec
            childrenvector = self.f.concatenate_with_bias(leftvector, rightvector)
            
            W_df = np.dot(delta_full, childrenvector.transpose())
            self.W_d += W_df
            V_df = self.get_tensor_gradient(delta_full, leftvector, rightvector)
            self.V_d += V_df
            
            delta_down = self.compute_tensor_delta_down(delta_full, leftvector, rightvector, self.model.W, self.model.V)
            
            left_derivative = self.f.element_wise_tanh_derivative(leftvector)
            right_derivative = self.f.element_wise_tanh_derivative(rightvector)
            left_delta_down = delta_down[0:delta_full.shape[0], 0:1]
            right_delta_down = delta_down[delta_full.shape[0]:delta_full.shape[0]*2, 0:1]
            
            self.backprop_derivatives_and_error(self, tree, tree.nodelist[cur_point].left, 
                                                self.f.element_mult(left_derivative, left_delta_down))
            self.backprop_derivatives_and_error(self, tree, tree.nodelist[cur_point].right,
                                                self.f.element_mult(right_derivative, right_delta_down))
        elif(tree.nodelist[cur_point].left):
            self.Ws_d += local_class_delta
            currentvector_derivative = self.f.element_wise_tanh_derivative(currentvector)
            delta_from_class = np.dot(self.model.Ws.transpose(), delta_class)
            delta_from_class = self.f.element_mult(delta_from_class[0:self.model.num_hid, 0:1], currentvector_derivative)
            delta_full = delta_from_class + delta
            leftvector = tree.nodelist[tree.nodelist[cur_point].left].nodevec
            
            delta_down = np.dot(self.model.W.transpose(), delta_full)
            
            left_derivative = self.f.element_wise_tanh_derivative(leftvector)
            left_delta_down = delta_down[0:delta_full.shape[0], 0:1]
            self.backprop_derivatives_and_error(self, tree, tree.nodelist[cur_point].left,
                                                self.f.element_mult(left_derivative, left_delta_down))
                                         
    def get_tensor_gradient(self, delta_full, lvec, rvec):
        s = delta_full.size
        V_df = np.zeros(shape=(2*s, 2*s, s), dtype=float)
        full_vector = self.f.concatenate(lvec, rvec)
        for i in range(s):
            # scale full_vector*delta_full[i][0]
            V_df[:,:,i] = np.dot(full_vector*delta_full[i][0], full_vector.transpose())
        return V_df

    def compute_tensor_delta_down(self, delta_full, lvec, rvec, w, v):
        w_trans_delta = np.dot(w.transpose(), delta_full)
        w_trans_delta_no_bias = w_trans_delta[0:w_trans_delta.shape[0]-1, 0:1]
        s = delta_full.size
        delta_tensor = np.zeros(shape=(2*s, 1), dtype=float)
        full_vector = self.f.concatenate(lvec, rvec)
        for i in range(size):
            scaled_full_vector = full_vector * delta_full[i][0]
            delta_tensor += np.dot((v[:,:,i]+v[:,:,i].transpose()), scaled_full_vector)
        return delta_tensor+w_trans_delta_no_bias

    def scale_and_regularize(self, derivatives, current_matrices, scale, reg_cost):
        cost = 0.0
        derivatives = derivatives*scale + current_matrices*reg_cost # scale
        cost += self.f.element_mult(current_matrices, current_matrices).sum() * reg_cost / 2.0
        return cost

    def scale_and_regularize4wordvec(self, derivatives, current_matrices, scale, reg_cost):
        cost = 0.0
        for w in current_matrices:
            d = derivatives[w]
            d = d*scale + current_matrices[w]*reg_cost # scale
            derivatives[w] = d
            cost +=  self.f.element_mult(current_matrices[w], current_matrices[w]).sum() * reg_cost / 2.0
        return cost
            
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
        new_array = np.zeros(shape=(new_array_row, 1), dtype=float)
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

    def element_wise_log(self, vec):
        out = np.zeros(shape=vec.shape, dtype=float)
        row, col = out.shape
        for i in range(row):
            for j in range(col):
                out[i][j] = math.log(vec[i][j])
        return out

    def element_mult(self, vec1, vec2):
        if vec1.shape != vec2.shape:
            raise ArraySizeError("array size unequal")
        col, row = vec1.shape
        out = np.zeros(shape=vec1.shape, dtype=float)
        for i in range(row):
            for j in range(col):
                out[i][j] = vec1[i][j] * vec2[i][j]
        return out
    
    def element_wise_tanh_derivative(self, vec):
        row, col = vec.shape
        out = np.ndarray(shape=(row, col), buffer=np.array([1.0]*(row*col)), dtype=float)
        out -= self.element_mult(vec, vec)
        return out
        
    def params2vector(self, total_size, W, V, Ws, L, model):
        theta = [0.0] * total_size
        i = 0
        try:
            for t in W.flat:
                theta[i] = t
                i += 1
            for t in V.flat:
                theta[i] = t
                i += 1
            for t in Ws.flat:
                theta[i] = t
                i += 1
            for w in model.wlist:
                for t in L[w].flat:
                    theta[i] = t
                    i+= 1
        except IndexError, e:
            sys.stderr.write("param2vector error, total_param_size is %d, current theta index is %d, error is %s \n"
                             % (total_size, i, str(e)))
            exit(1)
        if i != total_size:
            sys.stderr.write("param2vector error, total_param_size is %d, current theta index is %d \n",
                             %(len(theta), total_size))
            exit(1)
            
        return theta


class PipeLine:
    def __init__(self):
        pass
    
    def train(self, feature_file, dev_file, train_file, model_file):
        op = RntnOptions()
        m = RntnModel(feature_file)
        r = RNTN(op, m, feature_file, dev_file, train_file, model_file, None)
        r.load_data()
        r.train()
        r.save_model()

    def predict(self, feature_file, test_file, model_file, result_file):
        op = RntnOptions()
        m = RntnModel(feature_file)
        r = RNTN(op, m, feature_file, None, test_file, model_file, result_file)
        r.load_data()
        r.load_model()
        r.predict()


if __name__ == '__main__':
    feature_file = ''
    train_file = ''
    model_file = ''
    result_file = ''
    test_file = ''
    
    p = PipeLine()
    p.train(feature_file, None, train_file, model_file)
    p.predict(feature_file, test_file, model_file, result_file)
