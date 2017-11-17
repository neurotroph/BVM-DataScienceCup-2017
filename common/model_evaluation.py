#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Methods for evaluating our predictions using SMAPE.

Project:        BVM Data Science Cup 2017
Description:
    SMAPE was selected to evaluate the predictions for the first take - here we have a handy
    method to do that.
Author:         Christopher Harms
Email:          christopher.harms@skopos.de
Twitter:        @chrisharms

All rights reserved.

'''
import numpy as np


def SMAPE(predicted, true):
    return np.abs(predicted - true) / ((np.abs(true) + np.abs(predicted))/2.0)