import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    recall_score,
    precision_score,
    roc_auc_score,
    classification_report,
)

from src.logger import logger
from src.exception import CustomException
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join("artifacts", "fraud_model.pkl")


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig = ModelTrainerConfig()):
        self.config = config

    def evaluate_models(self, X_train, y_train, X_test, y_test, models: dict):
        results = []

        for name, model in models.items():
            logger.info(f"Training model: {name}")
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            recall = recall_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_prob)

            logger.info(f"{name} -> Recall: {recall:.4f}, Precision: {precision:.4f}, ROC-AUC: {roc_auc:.4f}")
            results.append({
                "name": name,
                "model": model,
                "recall": recall,
                "precision": precision,
                "roc_auc": roc_auc,
            })

        return results

    def initiate_model_trainer(self, train_arr: np.ndarray, test_arr: np.ndarray):
        try:
            logger.info("Starting model training")

             
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

            models = {
                "Logistic Regression": LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    n_jobs=-1
                ),
                "Random Forest": RandomForestClassifier(
                    n_estimators=150,
                    max_depth=12,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1
                ),
            }

            results = self.evaluate_models(X_train, y_train, X_test, y_test, models)

             
            best = max(results, key=lambda x: x["roc_auc"])
            best_model = best["model"]

            logger.info(
                f"Best model: {best['name']} | "
                f"Recall: {best['recall']:.4f}, "
                f"Precision: {best['precision']:.4f}, "
                f"ROC-AUC: {best['roc_auc']:.4f}"
            )
 
            y_pred_best = best_model.predict(X_test)
            logger.info("Classification report for best model:\n" +
                        classification_report(y_test, y_pred_best))

            # Save best model
            save_object(self.config.model_path, best_model)
            logger.info(f"Best model saved at: {self.config.model_path}")

            return self.config.model_path, best

        except Exception as e:
            logger.error("Error in ModelTrainer.initiate_model_trainer", exc_info=True)
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Optional quick end-to-end test
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation

    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    transformer = DataTransformation()
    train_arr, test_arr, prep_path = transformer.initiate_data_transformation(
        train_path, test_path
    )

    trainer = ModelTrainer()
    model_path, best = trainer.initiate_model_trainer(train_arr, test_arr)

    print("Model training complete!")
    print("Best model:", best["name"])
    print("Saved at:", model_path)
