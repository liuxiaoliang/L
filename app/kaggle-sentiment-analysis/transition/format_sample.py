#! /usr/bin/env python
# -*- encoding: utf-8 -*-

"""format sample

This module is used to format sample to normal format.
"""

__author__ = 'xiaoliang liu'

import sys

class FormatSample(object):
    """format sample
    
    Transform sample's format as below:
        sample_id lable f1:w1 f2:w2 ... # comment
    """
    
    def __init__(self, file_path_sample, file_output):
        self.file_path = file_path_sample
        self.file_new = file_output

    def format(self):
        """format

        Input format:
            sid label content
        Output format:
            sid label fea1:w1 fea2:w2
        """
        fp = open(self.file_path, 'r')
        fp2 = open(self.file_new, 'w')
        for line in fp:
            linelist = line.strip().split()
            try:
                sid = linelist[0]
                label = linelist[1]
                clist = linelist[2:]
            except IndexError, e:
                print("Unvalid line : {0}".format(line.strip()))
            cdict = {}
            for c in clist:
                cdict.setdefault(c, 0)
                cdict[c] += 1
            new_clist = [i[0] + ':' + str(i[1]) for i in cdict.items()]
            new_clist.insert(0, label)
            new_clist.insert(0, sid)
            fp2.write(' '.join(new_clist) + '\n')


if __name__ == '__main__':
    file_path = sys.argv[1]
    file_new = sys.argv[2]
    fs = FormatSample(file_path, file_new)
    fs.format()
            
