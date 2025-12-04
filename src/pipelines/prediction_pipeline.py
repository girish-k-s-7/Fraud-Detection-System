import sys
import pandas as pd
import numpy as np

from src.utils import load_object
from src.logger import logger
from src.exception import CustomException


class FraudPredictor:
    

    def __init__(
        self,
        model_path: str = "artifacts/fraud_model.pkl",
        preprocessor_path: str = "artifacts/preprocessor.pkl",
    ):
        try:
            logger.info("Loading preprocessor and model for prediction")
            self.model = load_object(model_path)
            self.preprocessor = load_object(preprocessor_path)
            logger.info("Loaded model and preprocessor successfully")
        except Exception as e:
            logger.error("Error loading model/preprocessor", exc_info=True)
            raise CustomException(e, sys)

    def predict(self, input_df: pd.DataFrame):
        
        try:
            logger.info(f"Running prediction on input of shape: {input_df.shape}")

            transformed = self.preprocessor.transform(input_df)
            preds = self.model.predict(transformed)
            probs = self.model.predict_proba(transformed)[:, 1]

            return preds, probs

        except Exception as e:
            logger.error("Error during prediction", exc_info=True)
            raise CustomException(e, sys)


if __name__ == "__main__":
    
    try:
        logger.info("Starting quick prediction test")

        test_df = pd.read_csv("artifacts/fraud_test.csv")
        feature_cols = [col for col in test_df.columns if col != "Class"]

        sample = test_df[feature_cols].head(5)

        predictor = FraudPredictor()
        preds, probs = predictor.predict(sample)

        print("Sample predictions:", preds)
        print("Fraud probabilities:", np.round(probs, 4))

    except Exception as e:
        raise CustomException(e, sys)
