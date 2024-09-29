import sys
from typing import Union
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from Vivi_AI.constant import *
from Vivi_AI.entity.config_entity import DataIngestionConfig, DataTransformationConfig
from Vivi_AI.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact, DataValidationArtifact
from Vivi_AI.exception import CustomException
from Vivi_AI.logger import logging
from sklearn.impute import SimpleImputer
from Vivi_AI.utils.main_utils import save_object, save_numpy_array_data, read_yaml, drop_columns

class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_transformation_config: DataTransformationConfig, data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self.schema_config = read_yaml(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformation_object(self) -> Pipeline:
        logging.info("Entered get_data_transformation_object method of DataTransformation class")

        try:
            logging.info("Fetching numerical, categorical, and transformation columns from schema config")
            columns_to_standardize = self.schema_config.get('Standard_columns', [])
            transform_columns = self.schema_config.get('Transformation_columns', [])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('standard', StandardScaler(), columns_to_standardize),
                    ('power', PowerTransformer(method='yeo-johnson', standardize=True), transform_columns),
                    # Assuming categorical columns need encoding

                ],
                remainder='passthrough'  # Keeps the columns not specified unchanged
            )

            logging.info("Created preprocessor object from ColumnTransformer")

            logging.info("Exited get_data_transformation_object method of DataTransformation class")
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Starting data transformation")
            preprocessor = self.get_data_transformation_object()
            logging.info("Preprocessor object obtained")

            drop_columns = self.schema_config.get('Drop_columns', [])
            train_df = self.read_data(file_path=self.data_ingestion_artifact.train_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info(f"Train df columns uis{train_df.columns}")
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN] + drop_columns, axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            logging.info(f"input feature columns is {input_feature_train_df.columns}")
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN] + drop_columns, axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]

            logging.info("Separated training and testing features and target")

            logging.info("Applying preprocessing to training and testing dataframes")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            logging.info("Preprocessing applied")

            logging.info("Applying SMOTE to the training dataset")
            smote = SMOTE(random_state=2)
            input_feature_train_final, target_feature_train_final = smote.fit_resample(input_feature_train_arr, target_feature_train_df)
            logging.info("SMOTE applied to the training dataset")

            logging.info("Transforming the test dataset without SMOTE")
            input_feature_test_final = preprocessor.transform(input_feature_test_df)
            logging.info("Test dataset transformation complete")

            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_df)]

            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            logging.info("Saved the preprocessor object and transformed data")
            # logging.info(f"traininhg data is {train_arr}")

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys) from e
