import pandas as pd
import os

from common import database
from models import item_sales
from fbprophet import Prophet

DATABASE_IP = "192.168.0.150"
ITEM_LIST = [34, 1230, 43, 28668, 29984]
PREDICTIONS_PATH = "data_predictions"

# Connect to database
try:
    db = database.db_connection(hostname=DATABASE_IP)
    if not db:
        raise Exception
except:
    print(f'''Could not connect to database at {DATABASE_IP}''')
    quit()

# (1) Get all average and maximum prices for Items
print("Get averaged data across all stations...\n")
sql = "SELECT typeid, time, AVG(avgprice) as avg_avgprice, MAX(maxprice) as max_maxprice FROM items WHERE " +\
      "typeid IN ({}) AND time >= '2015-12-01' GROUP BY typeid, time;".format(', '.join(map(str, ITEM_LIST)))
df_overall = pd.io.sql.read_sql(sql, db)

# (2) Get all stations from Items-Table that have sold our items in the timeframe
print("Get averages from stations...\n")
sql = "SELECT typeid, stationid, time, AVG(avgprice), MAX(maxprice), AVG(sales_units) AS avgsales FROM items WHERE typeid " +\
      "IN ({}) AND time >= '2015-12-01' ".format(', '.join(map(str, ITEM_LIST))) +\
      "GROUP BY typeid, stationid, time;"
df_stations = pd.io.sql.read_sql(sql, db)

#     Now join the two data frames and calculate differences
df_merged = pd.merge(df_stations, df_overall, on=["typeid", "time"])
df_merged["diff_avg"] = df_merged["avg"] - df_merged["avg_avgprice"]
df_merged["diff_max"] = df_merged["max"] - df_merged["max_maxprice"]
df_merged.to_csv(os.path.join(PREDICTIONS_PATH, "task-4_simple_all.csv"))
quit(0)

#     Aggregate, so we get average differences over the timeframe
df_sorted = df_merged.groupby(['typeid', 'stationid'])["diff_avg"].mean().groupby(level=0, group_keys=False)
#     Sort DataFrame by differences and output Top 10
df_topstations = df_sorted.nlargest(10)

#     Iterate through all items and
_all_predictions = pd.DataFrame()
_patch_dates = database.get_patch_dates(db)
for idx, type in enumerate(ITEM_LIST):
    print(f'''Getting historical data for all stations for Item #{type} ({idx+1}/{len(ITEM_LIST)})... \n''')
    # For each type and each station, get all historical data, train Prophet and predict next 39 days
    sql = "SELECT typeid, stationid, time, demand, dem_avg_p FROM supply_and_demand " + \
          "WHERE typeid = {} ".format(str(type)) + \
          "AND stationid IN ({}) ".format(', '.join(map(str, df_topstations[type].index)))
    df_item_stations = pd.io.sql.read_sql(sql, db)
    print(df_item_stations.head(10))

    for station in df_item_stations.stationid.unique():
        print(f'''Training Prophet for Item #{type} at Station #{station}... \n''')
        df = df_item_stations[(df_item_stations.stationid == station)].rename(index=str, columns={"time": "ds", "demand": "y"})
        #df = df.fillna(0)
        df["item"] = type
        predictor = item_sales.ItemSales()
        predictor.train(df)
        predictions = predictor.predict(periods=39, show_plots=False)
        predictions = predictions.rename(index=str, columns={"yhat": "dem_yhat"})
        predictions["type"] = type
        predictions["station"] = station
        _this_station = predictions

        df = df_item_stations[(df_item_stations.stationid == station)].rename(index=str, columns={"time": "ds", "dem_avg_p": "y"})
        #df = df.fillna(0)
        df["item"] = type
        predictor = item_sales.ItemSales()
        predictor.train(df)
        predictions = predictor.predict(periods=39, show_plots=False)
        predictions = predictions.rename(index=str, columns={"yhat": "price_yhat"})
        predictions["type"] = type
        predictions["station"] = station
        _all_predictions.append( pd.merge(_this_station, predictions, on=["type", "station", "ds"]) )

_all_predictions.to_csv(os.path.join(PREDICTIONS_PATH, "task-4_all.csv"))

db.close()

