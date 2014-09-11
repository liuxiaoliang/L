#! /usr/bin/env python
# -*- encoding: utf-8 -*-
#
# Copyright (c) <2014> <leon.max.liew@gmail.com>
# 

"""paragraph2vec

Implemention of the algorithm descripted in paper 
"Distributed Representations of Sentences and Documents".
"""
__author__ = "xiaoliang liu"

import sys
sys.path.append("../../common")
sys.path.append("../../mllib")

import math
import random
import pickle
import numpy as np
import data_prepare as dpp
