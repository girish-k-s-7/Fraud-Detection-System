import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer
from src.logger import logger
from src.exception import CustomException


def run_training_pipeline():
    try:
        logger.info("=== Training pipeline started ===")

        # 1. Data Ingestion
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()

        # 2. Data Transformation
        transformer = DataTransformation()
        train_arr, test_arr, preprocessor_path = transformer.initiate_data_transformation(
            train_path, test_path
        )

        # 3. Model Training
        trainer = ModelTrainer()
        model_path, best = trainer.initiate_model_trainer(train_arr, test_arr)

        logger.info("=== Training pipeline completed successfully ===")
        logger.info(f"Best model: {best['name']}")
        logger.info(f"Model saved at: {model_path}")
        logger.info(f"Preprocessor saved at: {preprocessor_path}")

        print("Training pipeline finished.")
        print("Best model:", best["name"])
        print("Model saved at:", model_path)
        print("Preprocessor at:", preprocessor_path)

    except Exception as e:
        logger.error("Error in training pipeline", exc_info=True)
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_training_pipeline()
