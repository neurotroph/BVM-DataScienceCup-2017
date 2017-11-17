#
# BVM Data Science Cup 2017
# Prediction Models for Item Sales
#
# Christopher Harms, SKOPOS GmbH, 2017
# christopher.harms@skopos.de
#
# All rights reserved.
#

import pandas as pd
import matplotlib.pyplot as plt
from fbprophet import Prophet

class ItemSales:
    """ Class to predict item sales. Basically a wrapper for Facebook Prophet. """

    def __init__(self, holiday_prior_scale=0.0, seasonality_prior_scale=10.0, changepoint_prior_scale=0.05,
                 changepoints=None, model='linear'):
        self._prior_scales = {'holiday': holiday_prior_scale,
                              'seasonality': seasonality_prior_scale,
                              'changepoints': changepoint_prior_scale}
        self._changepoints = changepoints

        if self._changepoints is not None:
            self._prophet = Prophet(growth=model,
                                    changepoint_prior_scale=self._prior_scales['changepoints'],
                                    changepoints=self._changepoints,
                                    n_changepoints=len(self._changepoints),
                                    weekly_seasonality=True,
                                    yearly_seasonality=True,
                                    seasonality_prior_scale=self._prior_scales['seasonality'],
                                    holidays=self.world_holidays(),
                                    holidays_prior_scale=self._prior_scales['holiday'])
        else:
            self._prophet = Prophet(growth=model,
                                    changepoint_prior_scale=self._prior_scales['changepoints'],
                                    weekly_seasonality=True,
                                    yearly_seasonality=True,
                                    seasonality_prior_scale=self._prior_scales['seasonality'],
                                    holidays=self.world_holidays(),
                                    holidays_prior_scale=self._prior_scales['holiday'])


    def world_holidays(self):
        christmas = pd.DataFrame({
            'holiday': 'christmas',
            'ds': pd.to_datetime(['2012-12-25', '2013-12-25', '2014-12-25', '2015-12-25', '2016-12-25', '2017-12-25']),
            'lower_window': 0,
            'upper_window': 1
        })
        newYear = pd.DataFrame({
            'holiday': 'new_year',
            'ds': pd.to_datetime(['2012-01-01', '2013-01-01', '2014-01-01', '2015-01-01', '2016-01-01', '2017-01-01']),
            'lower_window': 0,
            'upper_window': 0
        })
        laborday = pd.DataFrame({
            'holiday': 'laborday',
            'ds': pd.to_datetime(['2012-05-01', '2013-05-01', '2014-05-01', '2015-05-01', '2016-05-01', '2017-05-01']),
            'lower_window': 0,
            'upper_window': 0
        })
        easterMonday = pd.DataFrame({
            'holiday': 'easterMonday',
            'ds': pd.to_datetime(['2012-04-09', '2013-04-01', '2014-04-21', '2015-04-06', '2016-03-28', '2017-04-17']),
            'lower_window': -2,
            'upper_window': 0
        })
        return pd.concat((christmas, newYear, laborday, easterMonday))


    def train(self, training_data):
        if (training_data is None) or (type(training_data) is not pd.DataFrame):
            return False
        
        if (training_data['ds'].count() == 0) or (training_data['y'].count() == 0):
            return False

        # Remove changepoints that lie outside of training set (for whatever reason...)
        if (self._changepoints is not None) and (len(self._changepoints > 0)):
            self._changepoints = self._changepoints[(self._changepoints > training_data['ds'].min()) &
                                                    (self._changepoints < training_data['ds'].max())]
            self._prophet.changepoints = self._changepoints

        self._trained_data = training_data
        try:
            self._prophet.fit(self._trained_data)
        except:
            pass

        pass


    def validate(self, validation_data):
        if (validation_data is None) or (type(validation_data) is not pd.DataFrame):
            return False
        
        if (_trained_data['ds'].count() == 0) or (_trained_data['y'].count() == 0):
            return False

        # We will predict data outside the training data and return data with both,
        # predictions and real data

        predictions = self.predict(periods=(validation_data['ds'].max()-self._trained_data['ds'].max()).days)

        return pd.merge(validation_data, predictions, on="ds", how="right")


    def predict(self, periods, show_plots=False):
        if (type(periods) is not int) or (self._trained_data is None):
            return False
        
        future = self._prophet.make_future_dataframe(periods=periods)
        forecast = self._prophet.predict(future)
        # Add Item to DataFrame
        forecast['item'] = self._trained_data.item.unique()[0]

        if show_plots:
            self._prophet.plot_components(forecast)
            self._prophet.plot(forecast)
            plt.show()

        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'item']]


    @staticmethod
    def aggregate_to(period, prediction_data):
        if not (period == "weeks" or period == "months"):
            return False

        df = prediction_data
        df['year'] = df['ds'].map(lambda x: str(x.year))
        if period == "weeks":
            df['week'] = df['ds'].map(lambda x: str(x.isocalendar()[1]))
            return df.groupby(['year', 'week', 'item'], as_index=False)['yhat'].sum()

        if period == "months":
            df['month'] = df['ds'].map(lambda x: str(x.month))
            return df.groupby(['year', 'month', 'item'], as_index=False)['yhat'].sum()
