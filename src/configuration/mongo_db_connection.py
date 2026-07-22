import os
import sys
import certifi
import pymongo

from src.logger import logger
from src.constants import MONGODB_URL_KEY, DATABASE_NAME
from src.exception import CustomException

ca = certifi.where()
class MongoClient:
    client = None
    def __init__(self,database_name: str = DATABASE_NAME):
        try:
            if MongoClient.client is None:
                mongo_url = os.getenv(MONGODB_URL_KEY)
                if mongo_url is None:
                    raise CustomException("Mongo url is missing")
                MongoClient.client = pymongo.MongoClient(mongo_url, tls= True, tlsCAFile = ca)
            self.client = MongoClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
            logger.info("Mongo client has been created")
        except Exception as e:
            raise CustomException(e,sys)