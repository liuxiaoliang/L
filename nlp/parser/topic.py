#! /usr/bin/env python
# -*- encoding: utf-8 -*-
#
# Copyright (c) <2014> <leon.max.liew@gmail.com>
# 

"""topic model

Including lsa, plsa and lda.
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
