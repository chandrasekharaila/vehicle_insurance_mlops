import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logger


class TargetValueMapping:
    """Encapsulates binary target variable encoding and decoding."""
    
    def __init__(self):
        self.no: int = 0
        self.yes: int = 1

    def to_dict(self) -> dict:
        return {"no": self.no, "yes": self.yes}

    def reverse_mapping(self) -> dict:
        return {self.no: "no", self.yes: "yes"}


class MyModel:
    """Wrapper that applies feature preprocessing and model prediction."""

    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: pd.DataFrame) -> np.ndarray:
        """Transforms raw input dataframe and returns predictions array."""
        try:
            logger.info("Transforming input features and making predictions...")
            transformed_feature = self.preprocessing_object.transform(dataframe)
            return self.trained_model_object.predict(transformed_feature)

        except Exception as e:
            raise CustomException(e, sys) from e

    def __repr__(self) -> str:
        return f"MyModel(model={type(self.trained_model_object).__name__})"