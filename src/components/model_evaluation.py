import sys
from dataclasses import dataclass
from typing import Optional
import numpy as np
from sklearn.metrics import f1_score

from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
)
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.s3_estimator import Proj1Estimator
from src.exception import CustomException
from src.logger import logger
from src.utils.main_utils import load_numpy_array_data, load_object


@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: Optional[float]
    is_model_accepted: bool
    difference: float


class ModelEvaluation:
    def __init__(
        self,
        model_eval_config: ModelEvaluationConfig,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ):
        try:
            self.model_eval_config = model_eval_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys) from e

    def get_best_model(self):
        """Fetches active production model from S3 if present."""
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path = self.model_eval_config.s3_model_key_path
            proj1_estimator = Proj1Estimator(bucket_name=bucket_name, model_path=model_path)

            if proj1_estimator.is_model_present(model_path=model_path):
                return proj1_estimator
            return None
        except Exception as e:
            raise CustomException(e, sys) from e

    def evaluate_model(self) -> EvaluateModelResponse:
       
        try:
        
            test_arr = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_test_file_path
            )
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]

           
            trained_model = load_object(
                file_path=self.model_trainer_artifact.trained_model_file_path
            )
            y_pred_trained = trained_model.predict(x_test)
            trained_model_f1 = f1_score(y_test, y_pred_trained)
            logger.info(f"Newly Trained Model F1 Score: {trained_model_f1:.4f}")

           
            best_model_f1 = None
            production_model = self.get_best_model()

            if production_model is not None:
                logger.info("Evaluating current production model from S3...")
                y_pred_prod = production_model.predict(x_test)
                best_model_f1 = f1_score(y_test, y_pred_prod)
                logger.info(f"Production Model F1 Score: {best_model_f1:.4f}")

        
            prod_score = 0.0 if best_model_f1 is None else best_model_f1
            diff = trained_model_f1 - prod_score
            is_accepted = diff > 0

            response = EvaluateModelResponse(
                trained_model_f1_score=trained_model_f1,
                best_model_f1_score=best_model_f1,
                is_model_accepted=is_accepted,
                difference=diff,
            )
            logger.info(f"Evaluation Response: {response}")
            return response

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """Executes model evaluation component."""
        try:
            logger.info("Starting Model Evaluation Component.")
            eval_response = self.evaluate_model()

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=eval_response.is_model_accepted,
                s3_model_path=self.model_eval_config.s3_model_key_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=eval_response.difference,
            )

            logger.info(f"Model Evaluation Artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact

        except Exception as e:
            raise CustomException(e, sys) from e