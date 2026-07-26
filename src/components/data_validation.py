import os
import sys
import json

import pandas as pd
from pandas import DataFrame
from src.exception import CustomException
from src.logger import logger
from utils.main_utils import read_yaml_file

from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataValidationArtifact
from src.entity.artifact_entity import DataIngestionArtifact
from src.constants import SCHEMA_FILE_PATH

class DataValidation:
    def __init__(self,data_validation_config: DataValidationConfig, data_ingestion_artifact: DataIngestionArtifact):
        try:
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e,sys)

    
    def validate_number_of_columns(self,df: DataFrame):
        try:
            status = len(df.columns) == len(self._schema_config["columns"])
            if not status:
                logger.info("number of columns did not match")
            return status
        except Exception as e:
            raise CustomException(e,sys)


    
    def is_column_exist(self,df: DataFrame):
        try:
            schema_cols = self._schema_config["columns"]
            df_cols = df.columns
            cols_missing = False
            for col in schema_cols:
                if col not in df_cols:
                    logger.info(f"{col} is missing the dataframe")
                    cols_missing = True
                    break
                else:
                    logger.info(f"{col} is in the dataset")
            return cols_missing

        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_validation(self):
        try:
            logger.info("started data validation")
            validation_error_msg = ""
            train_df = pd.read_csv(self.data_ingestion_artifact.trained_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            status = DataValidation.validate_number_of_columns(train_df)
            if not status:
                validation_error_msg += "number of columns did not match in the training dataset"
            else:
                logger.info("Number of columns matched in the training dataset")

            status = DataValidation.validate_number_of_columns(test_df)
            if not status:
                validation_error_msg += "number of columns did not match in the testing dataset"
            else:
                logger.info("number of columns matched in the testing dataset")

            cols_missing = DataValidation.is_column_exist(train_df)
            if cols_missing:
                validation_error_msg += "cols are mssing in the training dataste"
                logger.info("cols are missing in the training dataset")
            else:
                logger.info("all cols are present in the training dataset")

            cols_missing = DataValidation.is_column_exist(test_df)
            if cols_missing:
                validation_error_msg += "cols are mssing in the testing dataset"
                logger.info("cols are missing in the testing dataset")
            else:
                logger.info("all cols are present in the testing dataset")

            validation_status = len(validation_error_msg)==0

            data_validation_artifact = DataValidationArtifact(
                validation_status= validation_status,
                message= validation_error_msg,
                validation_report_file_path= self.data_validation_config.validation_report_file_path
            )

            report_dir = os.path.dirname(self.data_validation_config.validation_report_file_path)
            os.makedirs(report_dir,exist_ok=True)

            validation_report = {
                "validation_status" : validation_status,
                "message": validation_error_msg
            }

            with open(self.data_validation_config.validation_report_file_path, "w") as report_file:
                json.dump(validation_report, report_file, indent=4)

            logger.info("Data Validation artifact created and saved to json file")
            return data_validation_artifact

        except Exception as e:
            raise CustomException(e,sys)