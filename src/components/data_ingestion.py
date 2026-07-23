import os
import sys
import numpy as np
import pandas as pd

from src.data_access.proj1_data import Proj1Data
from src.logger import logger
from src.exception import CustomException

from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataIngestionConfig

from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self,data_ingestion_config=DataIngestionConfig()):
        self.data_ingestion_config = data_ingestion_config

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logger.info("Data ingestion started")
            data = Proj1Data()
            df = data.export_data_as_dataframe(collection_name = self.data_ingestion_config.collection_name)
            logger.info("pulled data from database")

            dir_name = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(dir_name, exist_ok=True)
            df.to_csv(self.data_ingestion_config.feature_store_file_path, index= False, header = True)
            logger.info("data saved to the feature store")

            train_data, test_data = train_test_split(df, test_size=self.data_ingestion_config.train_test_split_ratio)
            logger.info("train test split done")

            dir_name = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_name, exist_ok=True)
            train_data.to_csv(self.data_ingestion_config.training_file_path,index= False, header=True)
            test_data.to_csv(self.data_ingestion_config.testing_file_path, index = False, header=True)
            logger.info("train and test data saved")

            data_ingestion_artifact = DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path, test_file_path=self.data_ingestion_config.testing_file_path)
            return data_ingestion_artifact


        except Exception as e:
            raise CustomException(e,sys)
        
