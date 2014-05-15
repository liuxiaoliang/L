#! /usr/bin/env python
#-*- encoding: utf-8 -*-
#
# Copyright (c) <2014> <leon.max.liew@gmail.com>
#
"""data preprocess

This module is used to preprocess input datas including
scaling, normlizing and so on. Now the input source is
only from file encoded in utf-8.
"""
__author__ = "xiaoliang liu"

import sys
import re

class Feature(object):
    def __init__(self):
        self.iid = 0 # int
        self.iweight = 0 # float

class Sample(object):
    def __init__(self):
        self.sid = None # str
        self.label = None # str
        self.flist = [] # feature list

class CateDPP(object):
    """data preprocess for classification

    Attributes:
        feature2id: a dict mapping feature to featre id
        id2feature: a dict mapping feature id to feature
        feature_num: total number of feature
        slist: a list storing sample matrix
        load: function for loading data
    """
    
    def __init__(self, file_path_feature, file_path_sample):
        """for labled data
        
        line in file_path_sample separated by a space:
        For example:
            sample_id lable f1:w1 f2:w2 ... # comment
            
        line in file_path_sample separated by a space with a head description:
        For example:
            ### desc
            Total number of feature : 134
            ###
            feautre_name feature_id
          
        """
        self.feature2id = {} # map feature to featre id
        self.id2feature = {} # map feature id to feature 
        self.feature_num = 0
        self.label_list = set()
        self.slist = [] # store sample 
        self.file_path_sample = file_path_sample
        self.file_path_feature = file_path_feature
    
    def load_feature(self):
        """get feture

        """
        fp = open(self.file_path_feature, 'r')
        lines = fp.readlines()
        fp.close()
        for l in lines:
            if l.find('###') >= 0:
                continue
            if l.find('Total number of feature') >= 0:
                self.feature_num = int(l.strip().split(':')[1].strip())
                continue
            
            try:
                (f, fid) = l.strip().split()
                fid = int(fid)
            except Exception, e:
                sys.stderr.write(str(e) + '\n')
                continue
            self.feature2id[f] = fid
            self.id2feature[fid] = f

    def load_sample(self):
        """load data

        """
        ret = 0
        try:
            fp = open(self.file_path_sample, 'r')
        except IOError, e:
            sys.stderr.write(str(e) + '\n')
            return ret
        line_no = 1
        for line in fp:
            if line[-1] != '\n':
                err_msg = "missing a newline character in the end"
                print("line {0}: {1}".format(line_no, err_msg))
                
            valid_part = line.split('#')[0].strip()
            if not valid_part:
                err_msg = "unvalid line"
                print("line {0}: {1}".format(line_no, err_msg))
                continue
            s = Sample()            
            nodelist = line.split()
            sid = nodelist.pop(0)
            s.sid = sid
            label = nodelist.pop(0)
            s.label = label
            for i in range(len(nodelist)):
                f = Feature()
                try:
                    fw = nodelist[i].split(':')
                    fea = ''
                    wei = 0
                    if len(fw) == 2:
                        fea, wei = fw
                    else: # feature has ':'
                        fea = ':'.join(fw[:-1])
                        wei = fw[-1]
                except Exception, e:
                    print("line {0}: feature format error {1} ".format(line_no, line.strip()))
                    continue
                try:
                    wei = float(wei)
                except ValueError, e:
                    print("line {0}: feature weight error {1} ".format(line_no, wei))
                    continue
                if fea not in self.feature2id:
                    continue
                f.iid = self.feature2id[fea]
                f.iweight = wei
                # get one feature
                s.flist.append(f)
            # get one sample
            if len(s.flist) > 0:
                self.slist.append(s)
            # next line
            line_no += 1

        fp.close()
    
    def normlizing(self):
        pass

    def scaling(self):
        pass

if __name__ == '__main__':
    file_path_feature = sys.argv[1]
    file_path_sample = sys.argv[2]
    cd = CateDPP(file_path_feature, file_path_sample)
    cd.load_feature()
    cd.load_sample()
    print cd.feature2id.items()[:3]
    print cd.slist[0].sid, cd.slist[0].label, [x.iid for x in cd.slist[0].flist]
    print cd.slist[1].sid, cd.slist[1].label, [x.iid for x in cd.slist[1].flist]
    
