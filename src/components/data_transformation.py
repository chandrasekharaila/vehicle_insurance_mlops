import sys
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import CustomException
from src.logger import logger
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_object(self) -> ColumnTransformer:
        
        try:
            logger.info("Initializing feature transformers.")

           
            num_features = self._schema_config.get('num_features', [])
            or_features = self._schema_config.get('ordinal_features', [])  # e.g., Gender
            oh_features = self._schema_config.get('one_hot_features', []) # e.g., Vehicle_Age, Vehicle_Damage

       
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

         
            ordinal_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ordinal_encoder', OrdinalEncoder())
            ])

       
            one_hot_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
            ])

          
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, num_features),
                    ("ordinal_pipeline", ordinal_pipeline, or_features),
                    ("one_hot_pipeline", one_hot_pipeline, oh_features)
                ],
                remainder='drop'  
            )

            logger.info("ColumnTransformer pipeline successfully created.")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logger.info("Data Transformation Started.")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

           
            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)

         
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]

       
            preprocessor = self.get_data_transformer_object()
            
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

           
            logger.info("Applying SMOTEENN to training dataset only.")
            smt = SMOTEENN(sampling_strategy="minority")
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                input_feature_train_arr, target_feature_train_df
            )

           
            input_feature_test_final = input_feature_test_arr
            target_feature_test_final = target_feature_test_df

           
            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]

            
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

            logger.info("Data Transformation completed successfully.")
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise CustomException(e, sys) from e