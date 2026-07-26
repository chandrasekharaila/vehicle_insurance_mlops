import sys
import numpy as np
from typing import Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logger
from src.utils.main_utils import load_numpy_array_data, load_object, save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact


class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    @staticmethod
    def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> ClassificationMetricArtifact:
        
        return ClassificationMetricArtifact(
            f1_score=f1_score(y_true, y_pred),
            precision_score=precision_score(y_true, y_pred),
            recall_score=recall_score(y_true, y_pred)
        )

    def train_model(self, x_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
        
        logger.info("Training RandomForestClassifier")
        
        model = RandomForestClassifier(
            n_estimators=self.model_trainer_config._n_estimators,
            min_samples_split=self.model_trainer_config._min_samples_split,
            min_samples_leaf=self.model_trainer_config._min_samples_leaf,
            max_depth=self.model_trainer_config._max_depth,
            criterion=self.model_trainer_config._criterion,
            random_state=self.model_trainer_config._random_state
        )
        
        model.fit(x_train, y_train)
        return model

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
       
        logger.info("Starting Model Trainer Component")
        try:
       
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)

            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            trained_model = self.train_model(x_train, y_train)

        
            y_pred = trained_model.predict(x_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            metric_artifact = self.evaluate_model(y_test, y_pred)

            logger.info(f"Test Accuracy: {test_accuracy:.4f} | F1: {metric_artifact.f1_score:.4f}")

            
            if test_accuracy < self.model_trainer_config.expected_accuracy:
                raise Exception(
                    f"Model performance ({test_accuracy:.4f}) is below expected threshold "
                    f"({self.model_trainer_config.expected_accuracy:.4f})"
                )

          
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            
            full_model_pipeline = Pipeline(steps=[
                ("preprocessor", preprocessing_obj),
                ("model", trained_model)
            ])

         
            save_object(self.model_trainer_config.trained_model_file_path, full_model_pipeline)
            logger.info(f"Full pipeline model successfully saved at: {self.model_trainer_config.trained_model_file_path}")

            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact
            )

        except Exception as e:
            raise CustomException(e, sys) from e