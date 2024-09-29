import json
import sys
import pandas as pd
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from pandas import DataFrame
from Vivi_AI.exception import CustomException
from Vivi_AI.logger import logging
from Vivi_AI.utils.main_utils import read_yaml, write_yaml_file
from Vivi_AI.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from Vivi_AI.entity.config_entity import DataValidationConfig
from Vivi_AI.constant import SCHEMA_FILE_PATH

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema_config = read_yaml(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e, sys)

    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        try:
            expected_columns = len(self.schema_config['columns'])
            actual_columns = len(dataframe.columns)
            status = actual_columns == expected_columns
            logging.info(f"Is required column present: [{status}]")
            return status
        except Exception as e:
            raise CustomException(e, sys)

    def is_column_exist(self, df: DataFrame) -> bool:
        try:
            dataframe_columns = df.columns
            missing_numerical_columns = [column for column in self.schema_config['numerical_columns'] if column not in dataframe_columns]
            missing_categorical_columns = [column for column in self.schema_config['categorical_column'] if column not in dataframe_columns]

            if missing_numerical_columns:
                logging.info(f"Missing numerical columns: {missing_numerical_columns}")

            if missing_categorical_columns:
                logging.info(f"Missing categorical columns: {missing_categorical_columns}")

            return not (missing_categorical_columns or missing_numerical_columns)
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys) from e

    def detect_dataset_drift(self, reference_df: DataFrame, current_df: DataFrame) -> bool:
        try:
            data_drift_profile = Profile(sections=[DataDriftProfileSection()])
            data_drift_profile.calculate(reference_df, current_df)
            report = data_drift_profile.json()

            json_report = json.loads(report)
            write_yaml_file(file_path=self.data_validation_config.drift_report_file_path, content=json_report)
            n_features = json_report['data_drift']['data']['metrics']['n_features']
            n_drifted_features = json_report['data_drift']['data']['metrics']['n_drifted_features']
            logging.info(f"{n_drifted_features}/{n_features} features drifted")
            drift_status = json_report['data_drift']['data']['metrics']['dataset_drift']
            return drift_status
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            validation_error_message = ""
            logging.info("Starting data validation")
            train_df = self.read_data(file_path=self.data_ingestion_artifact.train_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)

            status = self.validate_number_of_columns(dataframe=train_df)
            logging.info(f"All the required columns present in train dataframe: {status}")
            if not status:
                validation_error_message += 'Columns are missing in training dataframe. '

            status = self.validate_number_of_columns(dataframe=test_df)
            logging.info(f"All required columns present in testing dataframe: {status}")
            if not status:
                validation_error_message += "Columns are missing in test dataframe. "

            status = self.is_column_exist(df=train_df)
            logging.info(f"Column existence check for training data: {status}")
            if not status:
                validation_error_message += "Columns are missing in training dataframe. "

            status = self.is_column_exist(df=test_df)
            logging.info(f"Column existence check for testing data: {status}")
            if not status:
                validation_error_message += "Columns are missing in test dataframe. "

            validation_status = len(validation_error_message) == 0
            logging.info(f"Validation status is {validation_status}")

            if validation_status:
                drift_status = self.detect_dataset_drift(train_df, test_df)
                if drift_status:
                    logging.info("Drift Detected")
                    validation_error_message = "Drift Detected"
                else:
                    validation_error_message = "Drift not detected"
            else:
                logging.info(f"Validation error: {validation_error_message}")

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_message,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            logging.info(f"Data Validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise CustomException(e, sys) from e
