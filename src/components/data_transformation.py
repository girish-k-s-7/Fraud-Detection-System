import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.logger import logger
from src.exception import CustomException
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self, config: DataTransformationConfig = DataTransformationConfig()):
        self.config = config

    def get_preprocessor(self, feature_names):
        
        try:
            logger.info("Creating preprocessing pipeline")

            numeric_features = feature_names  

            preprocessor = ColumnTransformer(
                transformers=[
                    ("scaler", RobustScaler(), numeric_features)
                ]
            )

            return preprocessor

        except Exception as e:
            logger.error("Error in get_preprocessor", exc_info=True)
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            logger.info(f"Reading train data from: {train_path}")
            logger.info(f"Reading test data from: {test_path}")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

            # Separate features and target
            target_col = "Class"

            X_train = train_df.drop(columns=[target_col])
            y_train = train_df[target_col]

            X_test = test_df.drop(columns=[target_col])
            y_test = test_df[target_col]

            logger.info("Creating preprocessor with RobustScaler")
            preprocessor = self.get_preprocessor(feature_names=X_train.columns.tolist())

            logger.info("Fitting preprocessor on train data")
            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)

            logger.info("Saving preprocessor object")
            save_object(self.config.preprocessor_path, preprocessor)

            
            train_arr = np.c_[X_train_scaled, y_train.values]
            test_arr = np.c_[X_test_scaled, y_test.values]

            logger.info("Data transformation completed successfully")

            return train_arr, test_arr, self.config.preprocessor_path

        except Exception as e:
            logger.error("Error in DataTransformation.initiate_data_transformation", exc_info=True)
            raise CustomException(e, sys)


if __name__ == "__main__":
    
    from src.components.data_ingestion import DataIngestion

    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    transformer = DataTransformation()
    train_arr, test_arr, prep_path = transformer.initiate_data_transformation(
        train_path, test_path
    )

    print("Transformation complete!")
    print("Train array shape:", train_arr.shape)
    print("Test array shape:", test_arr.shape)
    print("Preprocessor saved at:", prep_path)
