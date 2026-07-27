import sys
import numpy as np
import pandas as pd

from src.cloud_storage.aws_storage import SimpleStorageService
from src.entity.estimator import MyModel
from src.exception import CustomException
from src.logger import logger


class Proj1Estimator:
    """Handles S3 model persistence and remote inference execution."""

    def __init__(self, bucket_name: str, model_path: str):
        self.bucket_name = bucket_name
        self.model_path = model_path
        self.s3 = SimpleStorageService()
        self.loaded_model: MyModel | None = None

    def is_model_present(self) -> bool:
        """Checks if the target model exists in S3."""
        try:
            return self.s3.s3_key_path_available(
                bucket_name=self.bucket_name, 
                s3_key=self.model_path
            )
        except Exception as e:
            logger.warning(f"Failed to check S3 model presence: {e}")
            return False

    def load_model(self) -> MyModel:
        """Downloads and loads the MyModel object from S3."""
        try:
            return self.s3.load_model(self.model_path, bucket_name=self.bucket_name)
        except Exception as e:
            raise CustomException(e, sys) from e

    def save_model(self, from_file: str, remove: bool = False) -> None:
        """Uploads a local model file to S3."""
        try:
            self.s3.upload_file(
                from_file,
                to_filename=self.model_path,
                bucket_name=self.bucket_name,
                remove=remove
            )
        except Exception as e:
            raise CustomException(e, sys) from e

    def predict(self, dataframe: pd.DataFrame) -> np.ndarray:
        """Executes prediction after lazy-loading the model from S3."""
        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model()
            return self.loaded_model.predict(dataframe=dataframe)
        except Exception as e:
            raise CustomException(e, sys) from e