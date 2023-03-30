import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    train_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Split training and test data")
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "LinearRegression": LinearRegression(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=False),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
            }

            ## Hyperparameter tuning for all the models

            params = {
                "LinearRegression": {},

                "DecisionTreeRegressor": {
                    "criterion": ["mse", "friedman_mse", "mae"],
                    "splitter": ["best", "random"],
                },

                # splitter : The strategy used to choose the split at each node. Supported strategies are “best” to 

                "KNeighborsRegressor": {
                    "n_neighbors": [3, 5, 7, 9]
                },

                "RandomForestRegressor": {
                    "n_estimators": [100, 200, 300, 400],
                    "criterion": ["mse", "mae"],
                },

                # criterion : The function to measure the quality of a split. Supported criteria are “friedman_mse” 
                # for the mean squared error with improvement score by Friedman, “mse” for mean squared error, and 
                # “mae” for the mean absolute error. The default value of “friedman_mse” is generally the best as it 
                # can provide a better approximation in some cases.

                "XGBRegressor": {
                    "n_estimators": [100, 200, 300, 400],
                    "learning_rate": [0.1, 0.2, 0.3, 0.4],
                },

                "CatBoostRegressor": {
                    "n_estimators": [100, 200, 300, 400],
                    "learning_rate": [0.1, 0.2, 0.3, 0.4],
                },

                "AdaBoostRegressor": {
                    "n_estimators": [100, 200, 300, 400],
                    "learning_rate": [0.1, 0.2, 0.3, 0.4],
                },

                "GradientBoostingRegressor": {
                    "n_estimators": [100, 200, 300, 400],
                    "learning_rate": [0.1, 0.2, 0.3, 0.4],
                }
                ## n_estimators is the number of trees in the forest
            }
            

            model_report: dict = evaluate_model(
                x_train = x_train,
                y_train = y_train,
                x_test = x_test,
                y_test = y_test,
                models = models,
                params=params
            )

            ## TO get the best model score from dictionary
            best_model_score = max(model_report.values(), key = lambda x: x["test_score"])

            ## TO get the best model name from dictionary
            best_model_name = [key for key, value in model_report.items() if value == best_model_score][0]

            if best_model_score.get("test_score") <= 0.6:
                raise CustomException("No best model found")
                
            logging.info(f"Best model is {best_model_name} with score {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=models[best_model_name]
            )

            predictiction = models[best_model_name].predict(x_test)
            score = r2_score(y_test, predictiction)
            return score

        except Exception as e:
            raise CustomException(e, sys)
