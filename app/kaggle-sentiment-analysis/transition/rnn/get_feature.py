#! /usr/bin/env python
# -*- encoding: utf-8 -*-

"""get feature

This module is used to preprocess sample to get kinds of feature.
"""

__author__ = "xiaoliang liu"

import sys

class GetFeature(object):
    """get feature

    feature file format:
    ### desc
    feature's total number: tnum
    ###
    f1 id1
    f2 id2
    ...

    
    Attributes:
        bow_feature : get bag-of-words feature
    
    """
    
    def __init__(self, file_path_sample, file_path_feature):
        
        self.file_path_sample = file_path_sample
        self.file_path_feature = file_path_feature
        
    def feature(self):
        """feature
        
        The input file is origin file.

        """
        feature2id = {}
        feature_id = 0
        fp = open(file_path_sample, 'r')
        for line in fp:
            linelist = line.strip().split('\t')
            try:
                wordlist =linelist[2].split()
            except IndexError, e:
                print("No conten in this line: {0}".format(line.strip()))
                continue
            for w in wordlist:
                if w not in feature2id:
                    feature2id[w] = feature_id
                    feature_id += 1
            
        fp.close()
        feature_id_list = feature2id.items()
        feature_id_list.sort(key = lambda x:x[1])
        fp = open(self.file_path_feature, 'w')
        fp.write("### DESC : feature\n" + 
                 "Total number of feature : " + str(len(feature_id_list)) + '\n'
                 "###\n")
        for f in feature_id_list:
            fp.write(f[0].ljust(10) + ' ' + str(f[1]).rjust(10) + '\n')
        fp.close()

if __name__ == '__main__':
    file_path_sample = sys.argv[1]
    file_path_feature = sys.argv[2]
    gf = GetFeature(file_path_sample, file_path_feature)
    gf.feature()
