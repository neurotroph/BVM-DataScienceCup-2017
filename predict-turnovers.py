import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from fbprophet import Prophet
from datetime import datetime

from common import database
from models import turnovers, evaluation

# Configuration
DATABASE_IP = "192.168.0.150"
PREDICTIONS_PATH = "data_predictions"
TURNOVER_DATA_PATH = "data_marketsegments"
SEGMENT_FILES_LIST = ["turnover_sliced_materials.csv",
                      "turnover_sliced_missile_launchers.csv",
                      "turnover_sliced_ships.csv"]

# Get patch dates for changepoints
with database.db_connection(hostname=DATABASE_IP) as db:
    _patchdates = database.get_patch_dates(db)

# Iterate through all three market segments
_all_predictions = pd.DataFrame()
for idx, segment_file in enumerate(SEGMENT_FILES_LIST):
    print(f'''Loading Data from file {segment_file} ({idx+1}/{len(SEGMENT_FILES_LIST)})...\n''')
    turnover_data = pd.read_csv(os.path.join(TURNOVER_DATA_PATH, segment_file), index_col=False, verbose=True)
    # Re-organize data
    turnover_data["y"] = np.log(turnover_data["y_raw"])
    # Now we need to iterate through all product groups in the data and predict each one
    product_groups = turnover_data['productgroup'].unique()
    for jdx, product_group in enumerate(product_groups):
        print(f'''Training model for Product Group "{product_group}" ({jdx+1}/{len(product_groups)})...\n''')
        predictor = turnovers.Turnovers(changepoints=_patchdates)
        predictor.train(turnover_data[(turnover_data.productgroup == product_group) & (turnover_data.y.notnull())])
        print(f'''Predicting the future for Product Group "{product_group}" ({jdx+1}/{len(product_groups)})...\n''')
        predictions_df = predictor.predict(4)
        predictions_df["segment"] = segment_file
        _all_predictions = _all_predictions.append(predictions_df, ignore_index=True)

# Save all predictions to file
_all_predictions.to_csv(os.path.join(PREDICTIONS_PATH, 'predictions_marketsegments_all_months.csv'))
print(f'''Written all predictions to disk...\n''')



