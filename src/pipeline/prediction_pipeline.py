import os
import sys
import pandas as pd
from src.exception.exception import customexception
from src.logger.logging import logging
from src.utils.utils import load_object


class PredictPipeline:
    def __init__(self):
        print("init.. the object")

    def predict(self, features):
        try:
            # directory where I want to create the folder
            parents_folder_path = r"C:\Users\yrobi\Desktop\Robin World\Data Science - Machine Learning Prep\01 - MLOps\MLOps-Project"
            artifacts_folder_path = os.path.join(parents_folder_path, "artifacts")
            preprocessor_path = os.path.join(artifacts_folder_path, "preprocessor.pkl")
            model_path = os.path.join(artifacts_folder_path,"model.pkl")

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            scaled_features = preprocessor.transform(features)
            pred = model.predict(scaled_features)

            return pred

        except Exception as e:
            raise customexception(e, sys)


class CustomData:
    def __init__(self,
                 carat:float,
                 depth:float,
                 table:float,
                 x:float,
                 y:float,
                 z:float,
                 cut:str,
                 color:str,
                 clarity:str):
        
        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity
            
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'carat':[self.carat],
                'depth':[self.depth],
                'table':[self.table],
                'x':[self.x],
                'y':[self.y],
                'z':[self.z],
                'cut':[self.cut],
                'color':[self.color],
                'clarity':[self.clarity]
                }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise customexception(e, sys)
        
obj = PredictPipeline()
        