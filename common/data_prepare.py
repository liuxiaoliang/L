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
            self.label_list.add(label)
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
            else:
                print("line {0}: feature empty ".format(line_no))
            # next line
            line_no += 1

        fp.close()
    
    def normlizing(self):
        pass

    def scaling(self):
        pass


class TreeNode(object):
    """tree node

    """
    def __init__(self, nid=None, nlabel=None, left=None,
                 right=None, parent=None, content=None):
        # tree arrtributes
        self.nid = nid # node id
        self.nlabel = nlabel
        self.left = left
        self.right = right
        self.parent = parent
        self.content = content
        self.fid = None # global feature id if this node is a singal word.
        self.leaf = False
        # rnn tree arrtributes
        self.prediction = None # node's cate predicted by model.
        self.nodevector = None # node's vector representation.
        self.index = -1 # indicate the index in prediction where cate is located in.
        self.error = 0 # error between real label and predicted label

    def isLeaf(self):
        return self.leaf
    
    def find(self, c):
        if self.content:
            return self.content.find(c)
        else:
            return -1

class RnnTree(object):
    """binary tree

    """
    def __init__(self):
        self.root = None
        self.size = 0
        self.nodelist = []
    
    def create(self, nodelist):
        """insert
        
        insert node by preorder traversal
        """
        if not nodelist:
            sys.stderr.write("empty nodelist\n")
            exit(0)
        self.root = 0
        self.size = len(nodelist)
        self.nodelist = nodelist
        parent_index = self.root
        s = []
        s.append(parent_index)
        p = 1
        while(len(s) > 0):
            # left children
            while(parent_index != None and 
                  p < self.size and
                  (not self.nodelist[parent_index].isLeaf()) and
                  self.nodelist[parent_index].find(self.nodelist[p].content) >= 0):
                # insert
                self.nodelist[parent_index].left = p
                self.nodelist[p].parent = parent_index 
                # go to left child
                parent_index = self.nodelist[parent_index].left
                s.append(parent_index)
                p += 1
                # is leaf, so don't go into left child
                if len(self.nodelist[parent_index].content.strip()) <=1:
                    self.nodelist[parent_index].leaf = True
                    break

            # right child
            if(len(s) > 0):
                parent_index = s.pop()
                if (p < self.size and
                    (not self.nodelist[parent_index].isLeaf()) and
                    self.nodelist[parent_index].find(self.nodelist[p].content) >= 0):
                    # insert
                    self.nodelist[parent_index].right = p
                    self.nodelist[p].parent = parent_index
                    # go to right child
                    parent_index = self.nodelist[parent_index].right
                    s.append(parent_index)
                    p += 1
                    # is leaf
                    if len(self.nodelist[parent_index].content.strip()) <=1:
                        self.nodelist[parent_index].leaf = True
                else:
                    # go to right child
                    parent_index = self.nodelist[parent_index].right
        
    def preorder_traversal(self):
        if self.root is None:
            print "Tree is empty"
            exit(0)
        p = self.root
        s = []
        while(len(s) > 0 or p != None):
            while(p != None):
                print self.nodelist[p].nid, self.nodelist[p].content, self.nodelist[p].nlabel
                if self.nodelist[p].fid:
                    print self.nodelist[p].fid
                s.append(p)
                p = self.nodelist[p].left
            if len(s) > 0:
                p = s.pop()
                p = self.nodelist[p].right
        
    def get_error_sum(self):
        error_sum = 0
        if self.root is None:
            print "Tree is empty"
            return error_sum
        
        p = self.root
        s = []
        while(len(s) > 0 or p != None):
            while(p != None):
                error_sum += self.nodelist[p].error
                s.append(p)
                p = self.nodelist[p].left
            if len(s) > 0:
                p = s.pop()
                p = self.nodelist[p].right
        
        return error_sum
        
    def get_predicted_label(self):
        r = []
        if self.root is None:
            print "Tree is empty"
            exit(0)
        p = self.root
        s = []
        while(len(s) > 0 or p != None):
            while(p != None):
                r.append((self.nodelist[p].nid, self.nodelist[p].index))
                s.append(p)
                p = self.nodelist[p].left
            if len(s) > 0:
                p = s.pop()
                p = self.nodelist[p].right
        return r

class RNNDPP(object):
    """data preprocess for treebank

    Attributes:
        feature2id: a dict mapping feature to featre id
        id2feature: a dict mapping feature id to feature
        feature_num: total number of feature
        slist: a list storing sample matrix
        load: function for loading data
    """

    def __init__(self, file_path_feature, file_path_sample):
        """for rnn data

        Every sample is a parser tree, as follow:
        sid1 sentence_id2 sentence label1 # separated by tab
        sid2 sentence_id2 sentence_part1 label2
        ...
        For example:
        156061  8545    An intermittently pleasing but mostly routine effort .  -1
        156062  8545    An intermittently pleasing but mostly routine effort    -1
        156063  8545    An      -1
        156064  8545    intermittently pleasing but mostly routine effort       -1
        156065  8545    intermittently pleasing but mostly routine      -1
        156066  8545    intermittently pleasing but     -1
        156067  8545    intermittently pleasing -1
        156068  8545    intermittently  -1
        156069  8545    pleasing        -1
        156070  8545    but     -1
        156071  8545    mostly routine  -1
        156072  8545    mostly  -1
        156073  8545    routine -1
        156074  8545    effort  -1
        156075  8545    .       -1
        """
        
        self.feature2id = {} # map feature to featre id
        self.id2feature = {} # map feature id to feature
        
        self.feature_num = 0
        self.label_list = set()
        self.slist = [] # store sample
        self.file_path_sample = file_path_sample
        self.file_path_feature = file_path_feature
    
    def load_feature(self):
        """get feature

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
        """load sample

        """
        fp = open(self.file_path_sample, 'r')
        sentence_id = -1
        partlist = []
        for line in fp:
            linelist = line.strip().split('\t')
            if len(linelist) < 4:
                sys.stderr.write("error line %s", line)
                continue
            sid = int(linelist[0])
            new_sentence_id = int(linelist[1])
            sent = linelist[2]
            label = int(linelist[3])
            if new_sentence_id != sentence_id:
                if partlist:
                    rnntree = RnnTree()
                    rnntree.create(partlist)
                    #rnntree.preorder_traversal()
                    self.slist.append(rnntree)
                partlist = []
                sentence_id = new_sentence_id
            t = TreeNode(nid=sid, nlabel=label, content=sent)
            if sent in self.feature2id:
                t.fid = self.feature2id[sent]
            partlist.append(t)
        # last part
        if partlist:
            rnntree = RnnTree()
            rnntree.create(partlist)
            #rnntree.preorder_traversal()
            self.slist.append(rnntree)
        fp.close()


class ConllDPP(object):
    """data preprocess for conll2007 dataset
    
    """
    START = ['-START-', '-START2-']
    END = ['-END-', '-END2-']
    def __init__(self, sample_file):
        self.sample_file = sample_file
        self.data4postagger = []
        self.data4parser = []

    def get_data(self):
        fp = open(self.sample_file, 'r')
        linelist = fp.read().strip().split('\n\n')
        for line in linelist:
            words = []
            tags = []
            heads = []
            labels = []
            tokenlist = line.split('\n')
            for t in tokenlist:
                tlist = t.strip().split('\t')
                try:
                    word = tlist[1]
                    tag = tlist[4]
                    head = tlist[6]
                    label = tlist[7]
                except:
                    continue
                words.append(word)
                tags.append(tag)
                heads.append((int(head) if head != '0' else len(tokenlist) + 1))
                labels.append(label)
            self._pad_tokens(words)
            self._pad_tokens(tags)
            self._pad_tokens(labels)
            heads.insert(0, 0)
            heads.append(-1)
            self.data4postagger.append((words, tags))
            self.data4parser.append((words, tags, heads, labels))
        fp.close()

    def _pad_tokens(self, tokens):
        tokens.insert(0, '<start>')
        tokens.append('ROOT')
        
    def normalize(self, word):
        if '-' in word and word[0] != '-':
            return '!HYPHEN'
        elif word.isdigit() and len(word) == 4:
            return '!YEAR'
        elif word[0].isdigit():
            return '!DIGITS'
        else:
            return word.lower()


class ClusterDPP(object):
    """csv dpp
    
    Process file for kmeans.
    Line in file_path_sample separated by a space:
    id fea1 fea2 ...
    
    Dense format.
    """
    def __init__(self, file_path):
        self._file_path = file_path
        self.slist = []
        self.dim = 0
    
    def load_sample(self):
        """load data

        """
        ret = 0
        try:
            fp = open(self._file_path, 'r')
        except IOError, e:
            sys.stderr.write(str(e) + '\n')
            return ret
        line_no = 1
        for line in fp:
            line = line.strip()
            linelist = line.split(' ')
            sid = linelist[0]
            flist = linelist[1:]
            s = Sample()
            s.sid = sid
            for f in flist:
                fe = Feature()
                if f:
                    fe.iweight = float(f)
                s.flist.append(fe)
            if len(s.flist) > 0:
                self.slist.append(s)
            line_no += 1
        # get dimension
        self.dim = len(self.slist[0].flist)
        
if __name__ == '__main__':
    file_path_feature = sys.argv[1]
    file_path_sample = sys.argv[2]
    #cd = CateDPP(file_path_feature, file_path_sample)
    #cd.load_feature()
    #cd.load_sample()
    #print cd.feature2id.items()[:3]
    #print cd.slist[0].sid, cd.slist[0].label, [x.iid for x in cd.slist[0].flist]
    #print cd.slist[1].sid, cd.slist[1].label, [x.iid for x in cd.slist[1].flist]
    #cd = RNNDPP(file_path_feature, file_path_sample)
    #cd.load_feature()
    #cd.load_sample()
    #print len(cd.slist)
    #cd = ConllDPP(file_path_sample)
    #cd.get_data()
    """
    for p in cd.data4parser:
        print '\t'.join(p[0])
        print '\t'.join(p[1])
        print '\t'.join([str(i) for i in p[2]])
        print '\t'.join(p[3])
        print
    """
    cd = ClusterDPP(file_path_sample)
    cd.load_sample()
    print cd.dim
    print len(cd.slist)
    for f in cd.slist[12].flist:
        print f.iweight
        
    
