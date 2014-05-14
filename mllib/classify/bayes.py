#! /usr/bin/env python
# -*- encoding: utf-8 -*-
#
# Copyright (c) <2014> <leon.max.liew@gmail.com>
# 

"""Bayes model

navie bayes
bayes network
"""

__author__ = 'xiaoliang liu'

import sys
sys.path.append("../../common/")
import data_prepare as dp

class NavieBayes(object):
    """nb
    
    Attributes:
        class_prior_prob: p(yj), prior probability.
        class_feature_prob_matrix: p(xi|yj).
        class_default_prob: default prob.
        train: method for training.
        predict: method for predicting.
    """
    
    class_prior_prob = {} # p(yj), prior probability
    class_feature_prob_matrix = {} # p(xi|yj)
    class_default_prob = {} # default prob
    laplace_lambda = 0.1 # lambda for laplace smooth
    
    def __init__(self, feature_file, sample_file, result_file, model_file=None):
        """input file
        
        Args:
            feature_file: feature file.
            sample_file: training or testing file.
            result_file: model file if training or predicting result if testing.
            model_file: only for predicting.
        """
        self.feature_file = feature_file
        self.sample_file = sample_file
        self.result_file = result_file
        self.model_file = model_file
        self.cdpp = dp.CateDPP(self.feature_file, self.sample_file)
        
    def train(self):
        """train

        """
        self.load_data()
        self.estimate_param()
        self.save_model()

    def predict(self):
        """test

        """
        pass

    def load_data(self):
        """data prepare

        """
        self.cdpp.load_feature()
        self.cdpp.load_sample()

    def estimate_param(self):
        """parameter estimation

        """
        class_count = {}
        class_feature_count = {}
        for s in self.cdpp.slist:
            class_count.setdefault(s.label, 0)
            class_count[s.label] += 1
            class_feature_count.setdefault(s.label, {})
            self.class_feature_prob_matrix.setdefault(s.label, {})
            for f in s.flist:
                class_feature_count[s.label].setdefault(f.iid, 0)
                class_feature_count[s.label][f.iid] += f.iweight
        # calculate p(yi)
        countY = float(sum(class_count.values()))
        for (k, v) in class_count.items():
            self.class_prior_prob[k] = v/countY
        # calculate p(xi|yj)
        for c in class_feature_count:
            countYj = float(sum(class_feature_count[c].values()) +
                            self.cdpp.feature_num) * self.laplace_lambda
            for f in class_feature_count[c]:
                self.class_feature_prob_matrix[c][f] = \
                    float(class_feature_count[c][f] + self.laplace_lambda) / countYj
            # default prob
            self.class_default_prob[c] = (float)(self.laplace_lambda) / countYj
            
    def save_model(self):
        """save model
        
        Model format as follow:
        ### p(yj)
        label1 label2 ...
        prob1 prob2 ...
        ### p(xi|yj)
        Feature label1 label2 ...
        f1 prob1 prob2 ...
        f2 prob1 prob2 ...
        
        """
        fp = open(self.result_file, 'w')
        fp.write('### p(yj)\n')
        fp.write(' '.join(self.cdpp.label_list) + '\n')
        probYj = [str(self.class_prior_prob[c]) for c in self.cdpp.label_list]
        fp.write(' '.join(probYj) + '\n')
        fp.write('### p(xi|yj)\n')
        fp.write('Feature ' + ' '.join(self.cdpp.label_list) + '\n')
        for f in self.cdpp.feature2id:
            fid = self.cdpp.feature2id[f]
            probXiYj = []
            for l in self.cdpp.label_list:
                if fid in self.class_feature_prob_matrix[l]:
                    probXiYj.append(str(self.class_feature_prob_matrix[l][fid]))
                else:
                    probXiYj.append(str(self.class_default_prob[l]))
            fp.write(f + ' ' + ' '.join(probXiYj) + '\n')
        fp.close()
        
    def load_model(self):
        """load model

        """
        pass

    def calculate(self):
        """predict posterier probability

        """
        pass

