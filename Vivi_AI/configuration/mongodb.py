import os 
import sys 
from Vivi_AI.constant import DB_NAME, CONNECTION_URL
import pymongo 
from Vivi_AI.logger import logging
from Vivi_AI.exception import CustomException

class MongoDBClient:
    client = None

    def __init__(self, database_name=DB_NAME) -> None:
        try:
            logging.info("Entering the MongoDB Client initialization.")
            
            # Check if client is already initialized
            if MongoDBClient.client is None:
                mongo_db_url = CONNECTION_URL
                if mongo_db_url is None or mongo_db_url.strip() == "":
                    raise ValueError("Environment key CONNECTION_URL is missing or empty.")
                
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url)
                logging.info("MongoDB client initialized successfully.")
            
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
            
            logging.info(f"Connected to MongoDB database: {database_name} successfully.")
        except Exception as e:
            logging.error(f"Error occurred while initializing MongoDB client: {e}")
            raise CustomException(e, sys) from e
