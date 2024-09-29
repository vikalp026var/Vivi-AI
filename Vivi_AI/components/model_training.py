import numpy as np
import pandas as pd
import autokeras as ak
import tensorflow as tf
import yaml
from sklearn.model_selection import KFold
from keras.utils import to_categorical
from Vivi_AI.constant import *
from Vivi_AI.exception import CustomException
from sklearn.preprocessing import LabelEncoder
from Vivi_AI.logger import logging
from Vivi_AI.utils.main_utils import load_numpy_array_data
from Vivi_AI.entity.config_entity import ModelTrainerConfig
from Vivi_AI.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
import sys
import os

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object(self, X_train, y_train, X_test, y_test):
        logging.info("Entering get_model_object method")
        try:
            logging.info("Initializing AutoKeras StructuredDataClassifier with max_trials=%d", MAX_TRAILS)
            clf = ak.StructuredDataClassifier(max_trials=MAX_TRAILS)
            logging.info("Starting model training with epochs=%d", EPOCHS)
            history = clf.fit(x=X_train, y=y_train, epochs=EPOCHS, validation_data=(X_test, y_test), callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE_EARLYSTOP, restore_best_weights=RESTORE_WEIGHTS),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=FACTOR, patience=PATIENCE_REDUCEONPLATEAU, min_lr=MIN_LR)
            ])
            logging.info("Model training completed successfully")
            model = clf.export_model()
            logging.info("Model exported successfully")

            # Save the model architecture
            model_path = self.model_trainer_config.trained_model_file_path
            model.save(model_path)
            logging.info(f"Model saved to {model_path}")

            # Save the model architecture to a YAML file
            model_yaml_path = self.model_trainer_config.model_config_file_path
            with open(model_yaml_path, 'w') as yaml_file:
                yaml_file.write(model.to_yaml())
            logging.info(f"Model architecture saved to {model_yaml_path}")

            return model
        except Exception as e:
            logging.error(f"Error in get_model_object: {e}")
            raise CustomException(e, sys)

    def KFold_Execute(self, X_train, y_train):
        logging.info("Entering KFold_Execute method")
        try:
            logging.info("Initializing KFold with n_splits=%d, shuffle=%s, random_state=%d", N_SPLITS, SHUFFLE, RANDOM_STATE)
            kf = KFold(n_splits=N_SPLITS, shuffle=SHUFFLE, random_state=RANDOM_STATE)
            acc_per_fold = []
            loss_per_fold = []
            fold_no = 1
            logging.info(type(X_train))
            logging.info(type(y_train))
            for train_index, test_index in kf.split(X_train):
                logging.info(f'Training fold {fold_no}...')
                
                # Convert indices to DataFrame slices
                X_train_fold = X_train.iloc[train_index]
                X_test_fold = X_train.iloc[test_index]
                y_train_fold = pd.Series(y_train[train_index])  # Convert to Series
                y_test_fold = pd.Series(y_train[test_index])   # Convert to Series

                # Convert labels to categorical
                num_classes = len(np.unique(y_train))
                y_train_fold = to_categorical(y_train_fold, num_classes=num_classes)
                y_test_fold = to_categorical(y_test_fold, num_classes=num_classes)

                logging.info("Getting model object for fold %d", fold_no)
                model = self.get_model_object(X_train_fold, y_train_fold, X_test_fold, y_test_fold)

                # Evaluate the model
                try:
                    logging.info("Evaluating model for fold %d", fold_no)
                    scores = model.evaluate(X_test_fold, y_test_fold, verbose=0)
                    logging.info(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')

                    acc_per_fold.append(scores[1] * 100)
                    loss_per_fold.append(scores[0])

                except Exception as e:
                    logging.error(f"Error during model evaluation: {e}")
                    raise

                fold_no += 1

            logging.info(f'Average accuracy: {np.mean(acc_per_fold):.2f}%')
            logging.info(f'Average loss: {np.mean(loss_per_fold):.4f}')
        except Exception as e:
            logging.error(f"Error in KFold_Execute: {e}")
            raise CustomException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entering initiate_model_trainer method")
        try:
            logging.info("Loading transformed train and test data from files")
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            logging.info("Data loaded successfully")
            
            # Convert to DataFrame for easy slicing
            X_train, y_train = pd.DataFrame(train_arr[:, :-1]), pd.Series(train_arr[:, -1])
            X_test, y_test = pd.DataFrame(test_arr[:, :-1]), pd.Series(test_arr[:, -1])
            X_train = X_train.astype('float32')
            X_test = X_test.astype('float32')

            # Encode labels
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)
            logging.info(f"Type of X_train {type(X_train)}")
            logging.info(f"Type of y_train {type(y_train)}")

            logging.info("Starting KFold execution")
            self.KFold_Execute(X_train, y_train)

            logging.info("Extracting the best model from final classifier")
            best_model = self.get_model_object(X_train, y_train, X_test, y_test)

            model_train_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                model_yaml_file_path=self.model_trainer_config.model_config_file_path
            )
            logging.info("ModelTrainerArtifact created successfully")
            return model_train_artifact
        except Exception as e:
            logging.error(f"Error in initiate_model_trainer: {e}")
            raise CustomException(e, sys)
