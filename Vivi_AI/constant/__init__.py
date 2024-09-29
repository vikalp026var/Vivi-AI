import os 
import sys 
from Vivi_AI.constant.env_variable import *
from datetime import date  




############## MongoDB Information ###############
DB_NAME="GBM_data"
COLLECTION_NAME="data"
CONNECTION_URL="mongodb+srv://vikalp026varshney:Vikalp026var@cluster0.r31hq0n.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"


############## COLUMN Information #################

TARGET_COLUMN="label"
PIPELINE_NAME:str="ViviAI"
ARTIFACT_DIR:str='artifact'

############## FILE Information ####################

FILE_NAME:str="GBM_data.csv"
TRAIN_FILE_NAME:str="train.csv"
TEST_FILE_NAME:str="test.csv"
PREPROCESSING_FILE_NAME:str="preprocessor.pkl"
MODEL_FILE_NAME="model.tf"
SCHEMA_FILE_PATH=os.path.join('config','schema.yaml')



############## Data Ingestion Contant ###############

DATa_INGESTION_COLLECTION_NAME:str="data"
DATA_INGESTION_DIR_NAME:str="data_ingestion"
DATA_INGESTION_FEATURE_SToRE_DIR:str="feature_store"
DATA_INGESTION_INGESTED_DIR:str="ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO:float=0.05


################ Data Validation Constant ##############


DATA_VALIDATION_DIR_NAME:str="data_validation"
DATA_VALIDATION_DRIFT_REPORT_DIR:str="drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME:str="report.yaml"



################# DATA TRANSFORMATION COnstant ###########

DATA_TRANSFORMATION_DIR_NAME:str="data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR:str="transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR:str="transformed_object"




###################MODEL Trainer COnstant ###################


MODEL_TRAINER_DIR_NAME:str="model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR:str="trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME:str="model.tf"
MODEL_TRAINER_EXPECTED_SCORE:float=0.8
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH:str=os.path.join("config","model.yaml")


######################## MODEL TRAIN CONFIG #################

EPOCHS=100
MAX_TRAILS=3
PATIENCE_EARLYSTOP=10
PATIENCE_REDUCEONPLATEAU=5
RESTORE_WEIGHTS=True
FACTOR=0.2
MIN_LR=0.001
N_SPLITS=5
SHUFFLE=True
RANDOM_STATE=42


#############
SUPABASE_URL="https://ucpbtniapifddpsipukq.supabase.co"
SUPABASE_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVjcGJ0bmlhcGlmZGRwc2lwdWtxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjM4MzE2NzQsImV4cCI6MjAzOTQwNzY3NH0.OScT9H0GnEsZYiV8wg2aBFwGAKUCRjOi9LhrzSn2R_w"



#######Google Auth #########
# {"web":{"client_id":"95250261471-o1qe4tsevfgga11d9bg3a6apneqv17kl.apps.googleusercontent.com","project_id":"vivoai","auth_uri":"https://accounts.google.com/o/oauth2/auth","token_uri":"https://oauth2.googleapis.com/token","auth_provider_x509_cert_url":"https://www.googleapis.com/oauth2/v1/certs","client_secret":"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","redirect_uris":["http://localhost:8080/signin-google"]}}


OAUTH2_CLIENT_ID="95250261471-o1qe4tsevfgga11d9bg3a6apneqv17kl.apps.googleusercontent.com"
OAUTH_SECRET_ID="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"
FLASK_SECRET="XwPp9xazJ0ku5CZnlmgAx2Dld8SHkAeT"
OAUTH2_META_URL="https://accounts.google.com/.well-known/openid-configuration"
