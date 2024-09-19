import sys,os
from dataclasses import dataclass

import numpy as np
import matplotlib as plt
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_filepath = os.path.join('artifacts',"preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config =DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        The function does the data transformation
        Preprocessing 
        '''
      
        try:
            numerical_cols = ["writing_score","reading_score"]
            categorical_cols = ["gender", "race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]
            
            num_pipeline = Pipeline(
                steps=[
                ("Imputer",SimpleImputer(strategy='median')),
                ("Scaler", StandardScaler())
                ])

            logging.info("Num Cols S_Scaling done!")

            cat_pipeline = Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy='most_frequent')),
                    ("One_hot_Encoder",OneHotEncoder()),
                    ("Standard_Scaling",StandardScaler(with_mean=False))
                ]
            )

            logging.info("Cat Cols Encoding done!")

            preprocessor = ColumnTransformer(  #Combining both pipelines!
                [
                    ("num_pipe",num_pipeline, numerical_cols),
                    ("cat_pipe", cat_pipeline, categorical_cols)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_trans(self, train_path, test_path):
        try:
            
            train_df = pd.read_csv(train_path)    
            test_df = pd.read_csv(test_path)    
            logging.info("Importing data for transformation- Data import complete")

            logging.info("Obtaining preprocessing object")
            
            preprocessing_obj = self.get_data_transformer_object()

            target_col_name = "math_score"
            numerical_cols = ["writing_score","readin_score"]

            input_feature_train_df = train_df.drop(columns = [target_col_name], axis=1)
            target_feature_train_df = train_df[target_col_name]

            input_feature_test_df = test_df.drop(columns=target_col_name, axis=1)
            target_feature_test_df = test_df[target_col_name]

            logging.info("Train and Test and Target Data ready!")
            logging.info(
                f"Applying preprocessing objects on training dataframe and testing dataframe"
            )    

            input_feature_train_array = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_array = preprocessing_obj.transform(input_feature_test_df)

            train_array = np.c_[
                input_feature_train_array, np.array(target_feature_train_df)
            ]
            test_array = np.c_[
                input_feature_test_array, np.array(target_feature_test_df)
            ]

            logging.info("Preprocessor object saved")

            save_object( #Creating the pkl file
                file_path = self.data_transformation_config.preprocessor_obj_filepath,
                obj=preprocessing_obj
            )

            return (
                train_array,
                test_array,
                self.data_transformation_config.preprocessor_obj_filepath
            )

        except Exception as e:
            raise CustomException(e, sys)













