#! /usr/bin/env python
# -*- encoding: utf-8 -*-
#
# Copyright (c) <2014> <leon.max.liew@gmail.com> 
#

"""linear programming

simple method
"""

__author__ = 'xiaoliang liu'

import os
import sys
import numpy as np

class SolverBase(object):
    """base

    """
    def __init__(self, ):
        pass

    def solve(self):
        pass

class SimplexByGauss(SolverBase):
    """gauss elimination
    
    """
    def __init__(self):
        SolverBase.__init__(self,)

    
class RevisedSimplex(SolverBase):
    """

    Use matrix product to solve linear programming question.
    """
    def __init__(self):
        SimplexBase.__init__(self,)

class DualSimplex(SolverBase):
    """
    
    Use dual reprsentation
    """
    def __init__(self):
        SolverBase.__init__(self,)

class InteriorPoint(SolverBase):
    pass

if __name__ == '__main__':
    print 'test'
