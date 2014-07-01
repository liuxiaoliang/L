#! /usr/bin/env python
#
# Copyright (c) <2014> <leon.max.liew@gmail.com>
#

"""common utilities

"""

__author__ = "xiaoliang liu"

import sys
import time

class Timing(object):
    """computer used time by a process
    
    unit: milliseconds
    """
    def __init__(self):
        self.cur_time = 0

    def start(self):
        self.cur_time = int(time.time()*1000)
        
    def report(self):
        return int(time.time()*1000) - self.cur_time

if __name__ == '__main__':
    t = Timing()
    t.start()
    time.sleep(2)
    print t.report()
