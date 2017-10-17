#!/usr/bin/env python

"""
    lr.py
    
    learning rate scheduler
"""

import numpy as np

class LRSchedule(object):
    
    @staticmethod
    def set_lr(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    @staticmethod
    def constant(x, lr_init=0.1, epochs=1):
        return lr_init
    
    @staticmethod
    def step(x, breaks=(150, 250)):
        if x < breaks[0]:
            return 0.1
        elif x < breaks[1]:
            return 0.01
        else:
            return 0.001
    
    @staticmethod
    def linear(x, lr_init=0.1, epochs=1):
        return lr_init * float(epochs - x) / epochs
    
    @staticmethod
    def cyclical(x, lr_init=0.1, epochs=1):
        """ Cyclical learning rate w/ annealing """
        if x < 1:
            # Start w/ small learning rate
            return 0.05
        else:
            return lr_init * (1 - x % 1) * (epochs - np.floor(x)) / epochs
