#! /usr/bin/env python
# -*- encoding: utf-8 -*-

# define some exception types


import sys

class ArraySizeError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)



if __name__ == '__main__':
    try:
        raise ArraySizeError("array size unequal")
    except Exception, e:
        print str(e)


    
