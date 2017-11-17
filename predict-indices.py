import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from fbprophet import Prophet
from datetime import datetime

from common import database
from models import indices, evaluation

# Configuration
DATABASE_IP = "192.168.0.150"
INDEX_LIST = ["CPI", "MPI", "SPPI"]
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
for idx, index in enumerate(INDEX_LIST):
    print(f'''Getting data for Index {index} ({idx+1} / {len(INDEX_LIST)})...\n''')
    predictor = indices.Indices(changepoints=_patchdates)
    index_df = database.get_price_index(index, db, True)
    print(f'''Training model for Index {index}...\n''')
    predictor.train(index_df)
    print(f'''Predicting the future for Index {index}...\n''')
    predictions_df = predictor.predict(100)
    _all_predictions = _all_predictions.append(predictions_df, ignore_index=True)

_all_predictions.to_csv(os.path.join(PREDICTIONS_PATH, 'predictions_indices_all_days.csv'))
print(f'''Predictions have been saved to predictions_indices_all.csv...\n''')

db.close()



