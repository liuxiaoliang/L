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
    def __init__(self, objective, constraint, isMin=True, iter_num = 0):
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
        self.iter_num = iter_num # iteration number
        
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
            coni = con2[i]
            coni.append(b[i])
            self.con_matrix.append(coni)
                        
    def solve(self):
        """
        
        Returns:
            A tuple including x and f(x).
        """
        return None

class SimplexByGauss(SolverBase):
    """gauss elimination
    
    """
    def __init__(self, obj, con, isMin, iter_num):
        SolverBase.__init__(self, obj, con, isMin, iter_num)
    
    def solve(self):
        self._standardize()
        for i in range(self.m):
            print self.con_matrix[i]
        self.__init_base_var()
        it = 0
        isHaveSolution = 1
        while True:
            it += 1
            print 'iter num: %d' % it
            self.__get_check_num()
            in_var = self.__get_in_var()
            if in_var == -1:
                break # get solution
            out_var = self.__get_out_var(in_var)
            if out_var == -1:
                isHaveSolution = 0
                break # no solution
            self.__exchange_in_out(in_var, out_var)
            if it > self.iter_num:
                break
        if isHaveSolution:
            print "x:"
            fx = 0
            for i, j in enumerate(self.base_var):
                print j, self.con_matrix[i][-1]
                fx += self.con_matrix[i][-1]*self.obj[j]
            if not self.isMin:
                fx = -1*fx
            print "f(x):", fx
        else:
            print 'no solution'

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
                    self.con_matrix[row][j], self.con_matrix[pi][j] = self.con_matrix[pi][j], self.con_matrix[row][j]
                self.con_matrix[row][self.n2], self.con_matrix[pi][self.n2] = self.con_matrix[pi][self.n2], self.con_matrix[row][self.n2] # b
            # elimination
            primary = self.con_matrix[row][col]
            for j in range(col, self.n2):
                self.con_matrix[row][j] /= float(primary)
            self.con_matrix[row][self.n2] /= float(primary) # b
            for i in range(0, self.m):
                if i == row:
                    continue
                pivot = self.con_matrix[i][col]
                for j in range(col, self.n2):
                    self.con_matrix[i][j] -= float(self.con_matrix[row][j]) * pivot
                self.con_matrix[i][self.n2] -= float(self.con_matrix[row][self.n2]) * pivot # b
            # next   
            self.base_var.append(col)
            row += 1
            col += 1

    def __get_check_num(self):
        self.check_num.clear()
        for j in range(self.n2):
            cp = 0
            if j not in self.base_var:
                for i in range(self.m):
                    cp += self.obj[self.base_var[i]]*self.con_matrix[i][j]
                cp = self.obj[j] - cp
                self.check_num[j] = cp
    
    def __get_in_var(self):
        res = -1
        __min = 0
        for k in self.check_num:
            tmp = self.check_num[k]
            if __min > tmp:
                __min = tmp
                res = k
        # if res == -1, then stop iterating.
        return res # res is the index of out variable

    def __get_out_var(self, in_var_index):
        res = -1
        a_jp = []
        for i in range(self.m):
            if self.con_matrix[i][in_var_index] > 0:
                a_jp.append((i, float(self.con_matrix[i][-1])/self.con_matrix[i][in_var_index]))
        if not a_jp:
            # no min value
            return res
        a_jp.sort(key=lambda x: x[1])
        res = a_jp[0][0]
        return res # base[res] is the index of out variable
        
    def __exchange_in_out(self, in_var_index, out_var_index):
        pivot = self.con_matrix[out_var_index][in_var_index]
        for i in range(self.n2):
            self.con_matrix[out_var_index][i] = self.con_matrix[out_var_index][i]/float(pivot)
        self.con_matrix[out_var_index][-1] = self.con_matrix[out_var_index][-1]/float(pivot) # b
        for i in range(self.m):
            if i == out_var_index:
                continue
            tmp = self.con_matrix[i][in_var_index]
            for j in range(self.n2):
                self.con_matrix[i][j] = self.con_matrix[i][j] - self.con_matrix[out_var_index][j]*tmp
            self.con_matrix[i][-1] = self.con_matrix[i][-1] - self.con_matrix[out_var_index][-1]*tmp # b
        # get new base_var
        self.base_var[out_var_index] = in_var_index

class RevisedSimplex(SolverBase):
    """

    Use matrix product to solve linear programming question.
    """
    def __init__(self, obj, con, isMin, iter_num):
        SolverBase.__init__(self, obj, con, isMin, iter_num)

class DualSimplex(SolverBase):
    """
    
    Use dual reprsentation
    """
    def __init__(self, obj, con, isMin, iter_num):
        SolverBase.__init__(self, obj, con, isMin, iter_num)

class InteriorPoint(SolverBase):
    pass

if __name__ == '__main__':
    obj = [-3, 2, -1, -1]
    cons = [[2, 1, 1, 3, '<=', 20],
            [1, 0, 1, 2, '=', 10],
            [-2, -1, 2, 5, '>=', 3]]
    print 'test SimplexByGauss'
    sg = SimplexByGauss(obj, cons, True, 100)
    sg.solve()
    
