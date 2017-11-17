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


class Turnovers(item_sales.ItemSales):
    """ Class to predict Indices. Identical to ItemSales"""

    def __init__(self, holiday_prior_scale=0.0, seasonality_prior_scale=10.0, changepoint_prior_scale=0.05,
                 changepoints=None):
        super(Turnovers, self).__init__(holiday_prior_scale,seasonality_prior_scale,changepoint_prior_scale,changepoints,
                                        model="linear")

    def predict(self, periods, show_plots=False):
        if (type(periods) is not int) or (self._trained_data is None):
            return False

        future = self._prophet.make_future_dataframe(periods=periods, freq="M")
        #future["cap"] = max(self._trained_data["cap"])
        forecast = self._prophet.predict(future)
        # Add additional data to dataframe
        forecast['productgroup'] = self._trained_data.productgroup.unique()[0]

        if show_plots:
            self._prophet.plot_components()
            self._prophet.plot()
            plt.show()

        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'productgroup']]
