import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score
from src.logger import logging
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(x_train, y_train, x_test, y_test, models, params):
    try:
        model_report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            param = list(params.values())[i]
            model_name = list(models.keys())[i]

            gs = GridSearchCV(model, param, cv=3)
            gs.fit(x_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train, y_train)

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            #log every model score
            logging.info(f"{model_name} train score: {train_model_score} test score: {test_model_score}")

            model_report[model_name] = {
                "train_score": train_model_score,
                "test_score": test_model_score,
            }
        return model_report
    
    except Exception as e:
        raise CustomException(e, sys)