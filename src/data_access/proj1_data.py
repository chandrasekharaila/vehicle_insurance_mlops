import sys
import pandas as pd
import numpy as np

from src.configuration.mongo_db_connection import MongoClient
from src.constants import DATABASE_NAME
from src.exception import CustomException
from src.logger import logger
from typing import Optional

class Proj1Data:

    def __init__(self):
        try:
            self.mongo_conn = MongoClient(database_name=DATABASE_NAME)
        except Exception as e:
            raise CustomException(e,sys)

    def export_data_as_dataframe(collection_name:str, database_name:Optional[str] = DATABASE_NAME):
        try:
            if database_name is None:
                collection = self.mongo_conn.database[collection_name]
            else:
                collection = self.mongo_conn.client[database_name][collection_name]

            data = list(collection.find())
            df = pd.DataFrame(data)

            if "id" in df.columns.to_list():
                df = df.drop("id", axis=1)

            df.fillna({"na":np.nan}, inplace=True)
            return df
        except Exception as e:
            raise CustomException(e,sys)
    
