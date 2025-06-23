import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method/component")
        try:
            # ✅ Step 1: Read dataset
            csv_path = 'C:/Users/rishi/mlproject1/notebook/data/stud.csv'
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"[ERROR] File not found: {csv_path}")

            df = pd.read_csv(csv_path)
            logging.info('✅ Successfully read the dataset')

            # ✅ Step 2: Create artifacts folder
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # ✅ Step 3: Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info('✅ Raw data saved at: ' + self.ingestion_config.raw_data_path)

            # ✅ Step 4: Split into train and test
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('✅ Train/test split done. Files saved.')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error("❌ Exception occurred during data ingestion", exc_info=True)
            raise CustomException(e, sys)


# ✅ Run this script directly
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    print(f"[SUCCESS] Train data saved at: {train_data}")
    print(f"[SUCCESS] Test data saved at: {test_data}")
    logging.info("Ingestion method called.")




    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))



