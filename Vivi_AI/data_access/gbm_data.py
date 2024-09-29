from Vivi_AI.configuration.mongodb import MongoDBClient
from Vivi_AI.constant import DB_NAME
from Vivi_AI.exception import CustomException
import pandas as pd
import sys
from typing import Optional

class GBM_Data:
    def __init__(self):
        try:
            self.mongo_client = MongoDBClient(database_name=DB_NAME)
        except Exception as e:
            raise CustomException(e, sys)

    def export_collection_as_dataframe(self, collection_name: str, database_name: Optional[str] = None) -> pd.DataFrame:
        try:
            # Access the specified collection
            if database_name:
                collection = self.mongo_client[database_name][collection_name]
            else:
                collection = self.mongo_client.database[collection_name]
            
            # Convert the collection to a DataFrame
            df = pd.DataFrame(list(collection.find()))
            
            # Drop the _id column if it exists
            if '_id' in df.columns:
                df = df.drop(columns=['_id'])
            
            return df
        except Exception as e:
            raise CustomException(e, sys) from e
