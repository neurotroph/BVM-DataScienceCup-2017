#
# BVM Data Science Cup 2017
# Prediction Models for Indices
#
# Christopher Harms, SKOPOS GmbH, 2017
# christopher.harms@skopos.de
#
# All rights reserved.
#

import pandas as pd
import matplotlib.pyplot as plt
from fbprophet import Prophet
from models import item_sales


class Indices(item_sales.ItemSales):
    """ Class to predict Indices. Identical to ItemSales"""

    def __init__(self, holiday_prior_scale=0.0, seasonality_prior_scale=10.0, changepoint_prior_scale=0.05,
                 changepoints=None):
        super(Indices, self).__init__(holiday_prior_scale,seasonality_prior_scale,changepoint_prior_scale,changepoints)
