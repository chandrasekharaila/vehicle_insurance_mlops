import sys

from src.cloud_storage.aws_storage import SimpleStorageService
from src.entity.artifact_entity import ModelEvaluationArtifact, ModelPusherArtifact
from src.entity.config_entity import ModelPusherConfig
from src.exception import CustomException
from src.logger import logger


class ModelPusher:
    def __init__(
        self,
        model_evaluation_artifact: ModelEvaluationArtifact,
        model_pusher_config: ModelPusherConfig,
    ):
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config
        self.s3_service = SimpleStorageService()

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """Uploads the newly trained model to S3 if it passed evaluation."""
        logger.info("Starting Model Pusher Component.")
        try:

            if not self.model_evaluation_artifact.is_model_accepted:
                logger.info("Trained model was NOT accepted. Skipping S3 upload.")
                raise Exception("Model evaluation failed. New model is worse than production. Pipeline aborted.")

           
            logger.info("Trained model accepted. Uploading to S3 bucket...")
            
            self.s3_service.upload_file(
                from_filename=self.model_evaluation_artifact.trained_model_path,
                to_filename=self.model_pusher_config.s3_model_key_path,
                bucket_name=self.model_pusher_config.bucket_name,
                remove=False
            )


            model_pusher_artifact = ModelPusherArtifact(
                bucket_name=self.model_pusher_config.bucket_name,
                s3_model_path=self.model_pusher_config.s3_model_key_path,
            )

            logger.info(f"Model successfully pushed to S3. Artifact: {model_pusher_artifact}")
            return model_pusher_artifact

        except Exception as e:
            raise CustomException(e, sys) from e