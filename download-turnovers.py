import os
from common import market_shares, database

# Configuration
DATABASE_IP = "192.168.0.150"
INDEX_LIST = ["CPI", "MPI", "SPPI"]
PREDICTIONS_PATH = "data_predictions"

TURNOVER_DATA_PATH = "data_marketsegments"

# Connect to database
try:
    db = database.db_connection(hostname=DATABASE_IP)
    if not db:
        raise Exception
except:
    print(f'''Could not connect to database at {DATABASE_IP}''')
    quit()

# Download all three market segments and store to disk
print("Downloading revenue turnover data for Ships...\n")
market_shares.download_tu_data("Ships", db).to_csv(os.path.join(TURNOVER_DATA_PATH, "turnover_ships.csv"))
print("Downloading revenue turnover data for Missile Launchers...\n")
market_shares.download_tu_data("Missile Launchers", db).to_csv(os.path.join(TURNOVER_DATA_PATH,
                                                                            "turnover_missile_launchers.csv"))
print("Downloading revenue turnover data for Materials...\n")
market_shares.download_tu_data("Materials", db).to_csv(os.path.join(TURNOVER_DATA_PATH, "turnover_materials.csv"))
