import os
import sys
import joblib

from src.logger import logger
from src.exception import CustomException


def save_object(file_path: str, obj) -> None:
     
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        joblib.dump(obj, file_path)
        logger.info(f"Object saved at: {file_path}")

    except Exception as e:
        logger.error("Error in save_object", exc_info=True)
        raise CustomException(e, sys)


def load_object(file_path: str):
     
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        obj = joblib.load(file_path)
        logger.info(f"Object loaded from: {file_path}")
        return obj

    except Exception as e:
        logger.error("Error in load_object", exc_info=True)
        raise CustomException(e, sys)
