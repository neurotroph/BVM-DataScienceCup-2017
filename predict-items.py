import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from fbprophet import Prophet
from datetime import datetime

from common import database
from models import item_sales, evaluation

# Configuration
DATABASE_IP = "192.168.0.150"
ITEM_LIST = [29984, 22548, 12005, 22544, 645,
             15681, 11578, 17559, 2281, 15806,
             28668, 30488, 32014, 2629, 32006,
             34, 36, 35, 38, 37]
PREDICTIONS_PATH = "data_predictions"

# Connect to database
try:
    db = database.db_connection(hostname=DATABASE_IP)
    if not db:
        raise Exception 
except:
    print(f'''Could not connect to database at {DATABASE_IP}''')
    quit()


dt = lambda x: datetime.strptime(x, '%Y-%m-%d').date()

# Get patch dates from server as changepoints
_patchdates = database.get_patch_dates(db)

# Retrieve predictions for items
_all_predictions = pd.DataFrame()
for idx, item in enumerate(ITEM_LIST):
    print(f'''Getting data for Item #{item} ({idx+1} / {len(ITEM_LIST)})...\n''')
    predictor = item_sales.ItemSales(changepoints=_patchdates)
    item_df = database.get_item_sales(item, db, False)
    print(f'''Training model for Item #{item}...\n''')
    predictor.train(item_df)
    print(f'''Predicting the future for Item #{item}...\n''')
    predictions_df = predictor.predict(100)
    _all_predictions = _all_predictions.append(predictions_df, ignore_index=True)

_all_predictions.to_csv(os.path.join(PREDICTIONS_PATH, 'predictions_items_all_days.csv'))
print(f'''Predictions have been saved to predictions_items_all.csv...\n''')

db.close()



