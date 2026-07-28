import sys
from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd

from src.entity.config_entity import VehiclePredictorConfig
from src.entity.s3_estimator import Proj1Estimator
from src.exception import CustomException
from src.logger import logger


@dataclass
class VehicleData:
    """Dataclass holding input features for vehicle prediction."""

    Gender: int | str
    Age: int
    Driving_License: int
    Region_Code: float
    Previously_Insured: int
    Annual_Premium: float
    Policy_Sales_Channel: float
    Vintage: int
    Vehicle_Age_lt_1_Year: int
    Vehicle_Age_gt_2_Years: int
    Vehicle_Damage_Yes: int

    def get_vehicle_input_data_frame(self) -> pd.DataFrame:
        """Converts the instance attributes into a single-row Pandas DataFrame."""
        try:
            logger.info("Converting VehicleData inputs to DataFrame...")
            # asdict turns dataclass attributes into a key-value dictionary instantly
            data_dict = {key: [value] for key, value in asdict(self).items()}
            return pd.DataFrame(data_dict)
        except Exception as e:
            raise CustomException(e, sys) from e


class VehicleDataClassifier:
    def __init__(self,prediction_pipeline_config: VehiclePredictorConfig = VehiclePredictorConfig()):
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
            
            self.estimator = Proj1Estimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys) from e

    def predict(self, dataframe: pd.DataFrame) -> np.ndarray:
        try:
            logger.info("Starting prediction using VehicleDataClassifier...")
            return self.estimator.predict(dataframe)
        except Exception as e:
            raise CustomException(e, sys) from e