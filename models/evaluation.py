#
# BVM Data Science Cup 2017
# Routines for Model Evaluation
#
# Christopher Harms, SKOPOS GmbH, 2017
# christopher.harms@skopos.de
#
# All rights reserved.
#

import numpy as np

def SMAPE_simple(predicted, true):
    return np.abs(predicted - true) / ((np.abs(true) + np.abs(predicted))/2.0)