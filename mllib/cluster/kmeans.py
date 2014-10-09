#! /usr/bin/env python
# -*- encoding: utf-8 -*-
#
# Copyright (c) <2014> <leon.max.liew@gmail.com>
#

"""kmeans

"""

__author__ = 'xiaoliang liu'

import sys
import os
from collections import defaultdict
from random import uniform
from math import sqrt
import time
import copy
sys.path.append("../../common/")
from data_prepare import *

class KMeans(object):
    """kmeans

    A simple cluster method
    """
    def __init__(self, sample_file, k_num, output_dir):
        self._filepath = sample_file
        self._k_num = k_num
        self._o_dir = output_dir
        self._dim = 0
        self._cdpp = ClusterDPP(self._filepath)
        self._cdpp.load_sample()
        self._dim = self._cdpp.dim
        self._centers = []
        self._clusters = []
        self._threshold = 0.00001

    def _random_k(self):
        """random

        Generate random K points as the initial cluster center.
        """
        min_max = defaultdict(float)

        for s in self._cdpp.slist:
            for i in xrange(self._dim):
                val = s.flist[i].iweight
                min_key = 'min_%d' % i
                max_key = 'max_%d' % i
                if min_key not in min_max or val < min_max[min_key]:
                    min_max[min_key] = val
                if max_key not in min_max or val > min_max[max_key]:
                    min_max[max_key] = val

        for k in xrange(self._k_num):
            s = Sample()
            for i in xrange(self._dim):
                min_val = min_max['min_%d' % i]
                max_val = min_max['max_%d' % i]

                fe = Feature()
                fe.iweight = uniform(min_val, max_val)
                s.flist.append(fe)

            self._centers.append(s)

    def _clusting(self):
        """clusting

        Assign each point to an index that corresponds to the index
        of the center point on it's proximity to that point.
        """
        self._clusters = []
        for s in self._cdpp.slist:
            shortest = ()  # positive infinity
            shortest_index = 0
            for i in xrange(len(self._centers)):
                val = self._distance(s, self._centers[i])
                if val < shortest:
                    shortest = val
                    shortest_index = i
            self._clusters.append(shortest_index)

    def _update(self):
        """update center

        """
        new_means = defaultdict(list)
        for c, s in zip(self._clusters, self._cdpp.slist):
            new_means[c].append(s)

        for cnt, clt in new_means.items():
            self._centers[cnt] = self._cmpt_cnt(clt)

    def _distance(self, a, b):
        """distance

        Computer distance between a and b.

        Args:
            a, b: Sample object.

        Returns:
            float type.

        """
        _sum = 0
        if (not a) or (not b):
            return 1
        for d in xrange(self._dim):
            difference_sq = (a.flist[d].iweight - b.flist[d].iweight) ** 2
            _sum += difference_sq
        return sqrt(_sum)

    def _cmpt_cnt(self, points):
        """computer center

        Args:
            points: a Sample object list.

        Returns:
            center: a Sample object.

        """
        new_center = Sample()

        for d in xrange(self._dim):
            dim_sum = 0  # dimension sum
            for p in points:
                dim_sum += p.flist[d].iweight
            # average of each dimension
            fe = Feature()
            fe.iweight = dim_sum / float(len(points))
            new_center.flist.append(fe)

        return new_center

    def kmeans(self):
        """kemans

        """
        self._random_k()
        self._clusting()
        iter_num = 1
        while True:
            print 'iter num: %d' % iter_num
            sys.stdout.flush()
            iter_num += 1
            old_centers = copy.deepcopy(self._centers)
            self._update()
            self._clusting()
            # determine whether to stop iterating
            centers_sum = 0
            for i in xrange(len(self._centers)):
                centers_sum += self._distance(old_centers[i], self._centers[i])
            print centers_sum
            if centers_sum < self._threshold:
                break
            time.sleep(1)

    def save(self):
        if not os.path.exists(self._o_dir):
            os.mkdir(self._o_dir)
        clusters = defaultdict(list)
        for c, s in zip(self._clusters, self._cdpp.slist):
            clusters[c].append(s.sid)
        for c in clusters:
            cfp = open(self._o_dir + '/' + str(c), 'w')
            for sid in clusters[c]:
                cfp.write(sid + '\n')
            cfp.close()


if __name__ == '__main__':
    k = KMeans('../../dataset/cluster/kmeans_sample2', 100, './test')
    k.kmeans()
    k.save()
