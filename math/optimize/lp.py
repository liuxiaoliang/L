#! /usr/bin/env python
# -*- encoding: utf-8 -*-
#
# Copyright (c) <2014> <leon.max.liew@gmail.com> 
#

"""linear programming

1) simplex
2) interior point
"""

__author__ = 'xiaoliang liu'

import os
import sys
import numpy as np

class SolverBase(object):
    """

    Slove programming question with standardized expression as follows:
        min  f(x) = C.T*X
        s.t. A*x = b, x>=0
    So, we should standardize the constraint equations first.
    We won't care about case when there is no solution about A*x = b.
    """
    def __init__(self, objective, constraint, isMin=True):
        """
        
        Args:
            objective: A array including C.T.
            constraint: A array including A and b before standardization.
            isMin: whether to minimize.
            For example:
            max  2x1 + 3x2
            s.t. -x1 + x2 <= 3
                 -2x1 + x2 <= 2
                 -4x1 - x2 >= -16
                 x1 >= 0, x2 >= 0
            so objective is [2, 3] and constraint is
            [[-1, 1, ,'<=', 3], [-2, 1, '<=', 2], [-4, -1, '>=', -16]]
            
        """
        self.obj = objective
        self.con = constraint
        self.con_matrix = [] # constraint matrix after standardization.
        self.isMin = isMin
        self.base_var = [] # index of basic variable
        self.check_num = {} # check number
        self.n = len(objective) # number of variable
        self.m = len(constraint) # number of constraint
        self.n2 = self.n # number of variable after standardization
        
    def _standardize(self):
        con2 = [a[:-2] for a in self.con]
        b = [a[-1] for a in self.con]
        if not self.isMin:
            self.obj = [ a*-1 for a in self.obj]
        for i in range(self.m):
            sign = self.con[i][-2]
            if sign == '<' or sign == '<=':
                self.obj.append(0)
                self.n2 += 1
                for j in range(self.m):
                    if i == j:
                        con2[j].append(1)
                    else:
                        con2[j].append(0)
            elif sign == '=':
                pass
            else:
                con2[i] = [-1*a for a in con2[i]]
                b[i] *= -1
                self.obj.append(0)
                self.n2 += 1
                for j in range(self.m):
                    if i == j:
                        con2[j].append(1)
                    else:
                        con2[j].append(0)
        for i in range(self.m):
            self.con_matrix.append(con2[i].append(b[i]))
                        
    def solve(self):
        """
        
        Returns:
            A tuple including x and f(x).
        """
        return None

class SimplexByGauss(SolverBase):
    """gauss elimination
    
    """
    def __init__(self, obj, con, isMin):
        SolverBase.__init__(self, obj, con, isMin)
    
    def solve(self):
        self._standardize()
        
    def __init_base_var(self):
        """

        Initialize base variable by Gaussian elimination.
        Won't care about case when there is no solution about A*x = b.
        """
        row = 0;col = 0;
        while (row < self.m and col < self.n2):
            # find primary
            p = -1
            pi = -1
            for r in range(row, self.m):
                if(abs(self.con_matrix[r][col]) > p):  
                    p = abs( self.con_matrix[r][col])  
                    pi = r
            if(float(p) == 0.0):  
                col += 1  
                continue
            # swap tow rows
            if pi != row:  
                for j in range(col, self.n2):  
                    swap(self.con_matrix[row][j], self.con_matrix[pi][j] )  
                swap(self.con_matrix[row][self.n2], self.con_matrix[pi][self.n2]) # b
            # elimination
            for j in range(col, self.n2):
                self.con_matrix[row][j] /= self.con_matrix[row][col]
            self.con_matrix[row][self.n2] /= self.con_matrix[row][self.n2] # b
            for i in range(0, self.m):
                if i == row:
                    continue
                for j in range(col, self.n2):
                    self.con_matrix[i][j] -= self.con_matrix[row][j] * self.con_matrix[i][col]
                self.con_matrix[i][self.n2] -= self.con_matrix[row][self.n2] * self.con_matrix[i][col] # b
            # next   
            self.base_var.append(col)
            row += 1
            col += 1

    def __get_check_num(self):
        self.check_num.clear()
        for j in range(self.n2):
            cp = 0
            if j not in self.base_var:
                for i in range(m):
                    cp += self.obj[self.base_var[i]]*self.con_matrix[i][j]
                cp = self.obj[j] - cp
                self.check_num[j] = cp
    
    def __get_in_var(self):
        pass

    def __get_out_var(self):
        pass
    
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
