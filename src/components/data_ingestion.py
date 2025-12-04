import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.logger import logger
from src.exception import CustomException


class DataIngestion:
    def __init__(self):
        
        self.train_data_path = os.path.join("artifacts", "fraud_train.csv")
        self.test_data_path = os.path.join("artifacts", "fraud_test.csv")

        
        self.source_path = r"Notebooks/data/processed/cleaned_fraud.csv"
      

    def initiate_data_ingestion(self):
        logger.info("Starting data ingestion process")

        try:
            logger.info(f"Reading source data from: {self.source_path}")
            df = pd.read_csv(self.source_path)
            logger.info(f"Source data shape: {df.shape}")

            #  drop duplicates again for safety
            dup_before = df.duplicated().sum()
            df = df.drop_duplicates()
            dup_after = df.duplicated().sum()
            logger.info(f"Dropped {dup_before - dup_after} duplicate rows")

            # Split into features and target
            X = df.drop(columns=["Class"])
            y = df["Class"]

            logger.info("Performing stratified train-test split")
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                stratify=y,
                random_state=42
            )

            train_df = pd.concat([X_train, y_train], axis=1)
            test_df = pd.concat([X_test, y_test], axis=1)

            os.makedirs("artifacts", exist_ok=True)

            train_df.to_csv(self.train_data_path, index=False)
            test_df.to_csv(self.test_data_path, index=False)

            logger.info(f"Train data saved at {self.train_data_path}, shape: {train_df.shape}")
            logger.info(f"Test data saved at {self.test_data_path}, shape: {test_df.shape}")
            logger.info("Data ingestion completed successfully")

            return self.train_data_path, self.test_data_path

        except Exception as e:
            logger.error("Error during data ingestion", exc_info=True)
            raise CustomException(e, sys)


if __name__ == "__main__":
    ingestor = DataIngestion()
    train_path, test_path = ingestor.initiate_data_ingestion()
    print("Ingestion complete!")
    print("Train path:", train_path)
    print("Test path:", test_path)
