import warnings
import pandas as pd
import numpy as np
import csv
#from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score,mean_squared_error,mean_absolute_error, r2_score
import matplotlib.pyplot as plt
#from pandas_profiling import ProfileReport
import seaborn as sns
import pickle
from urllib.parse import urlparse
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

import logging
import mlflow
import mlflow.sklearn

logging.basicConfig(filename='arem_log_logireg.log', encoding='utf-8', level=logging.INFO)
#logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def process_files(folderpath):
    # os.getcwd() + "\AReM"  # get present working directory location here
    location = folderpath
    counter = 0  # keep a count of all files found
    # csvfiles = [] #list to store all csv files found at location
    total_file_cnt = 0
    df_all = pd.DataFrame()

    for fol in next(os.walk('AReM'))[1]:
        logger.info("Folder Name:\t", fol)
        print(os.getcwd() + "\AReM" + "\\" + fol)
        counter = 0
        for file in os.listdir(os.getcwd() + "\AReM" + "\\" + fol):
            filepath = ""
            try:
                if file.endswith(".csv"):
                    filepath = os.getcwd() + "\AReM" + "\\" + fol + "\\" + file
                    #print("csv file found:\t", filepath)
                    logger.info("csv file found:\t", filepath)
                    # csvfiles.append(str(file))
                    df = pd.read_csv(filepath, skiprows=4, encoding='utf-8',
                                     error_bad_lines=False, quoting=csv.QUOTE_NONE)
                    df["target"] = fol.replace("1", "").replace("2", "")
                    df_all = df_all.append(df)
                    # print ("df shape:\t", df.shape)
                    counter = counter+1
                    total_file_cnt = total_file_cnt + 1
            except Exception as e:
                logger.exception(e)
                logger.exception("No files found here!")

        logger.info(f"Total files in {fol} folder found:{counter}".format(
            fol=fol, counter=counter))
    logger.info("Total files found:\t", total_file_cnt)
    return df_all


def mlflow_experiment_name(name):
    name = name + "-exp"
    # set experiment name to organize runs
    mlflow.set_experiment(name)
    experiment = mlflow.get_experiment_by_name(name)
    return experiment


def handling_null_zeros(df_all):
    try:        
        df_all['avg_rss12'].fillna(df_all['avg_rss12'].mean(), inplace=True)
        df_all['avg_rss12'] = df_all['avg_rss12'].replace(
            0, df_all['avg_rss12'].mean())

        df_all['var_rss12'].fillna(df_all['var_rss12'].mean(), inplace=True)
        df_all['var_rss12'] = df_all['var_rss12'].replace(
            0, df_all['var_rss12'].mean())

        df_all['avg_rss13'].fillna(df_all['avg_rss13'].mean(), inplace=True)
        df_all['avg_rss13'] = df_all['avg_rss13'].replace(
            0, df_all['avg_rss13'].mean())

        df_all['var_rss13'].fillna(df_all['var_rss13'].mean(), inplace=True)
        df_all['var_rss13'] = df_all['var_rss13'].replace(
            0, df_all['var_rss13'].mean())

        df_all['avg_rss23'] = df_all['avg_rss23'].replace(
            0, df_all['avg_rss23'].mean())
        df_all['avg_rss23'].fillna(df_all['avg_rss23'].mean(), inplace=True)

        df_all['var_rss23'] = df_all['var_rss23'].replace(
            0, df_all['var_rss23'].mean())
        df_all['var_rss23'].fillna(df_all['var_rss23'].mean(), inplace=True)

    except Exception as e:
        logger.exception("Error in handling the nulls: %s", e)


def data_preprocess():
    try:
      df = process_files(os.getcwd() + "\AReM")
    except Exception as e:
        logger.exception("Error in pre process files: %s", e)

    try:
      handling_null_zeros(df)
    except Exception as e:
        logger.exception("Error while handling Null and Zeros: %s", e)

    df.drop("# Columns: time", axis=1, inplace=True)
    logger.info("# Columns: time dropped.")

    try:
        enc = LabelEncoder()
        y = enc.fit_transform(df.target)
        x = df.drop(columns=['target'])

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=144, stratify=y)

        logger.info('x_train.shape', x_train.shape)
        logger.info('x_test.shape', x_test.shape)
        logger.info('y_train.shape', y_train.shape)
        logger.info('y_test.shape', y_test.shape)

        return x_train, x_test, y_train, y_test

    except Exception as e:
        logger.exception("Error while splitting the df: %s", e)


# X_train, X_test, y_train, y_test 
# LogisticRegression

def LogisticRegression_exp(exp_name,x_train,x_test,y_train,y_test):
    experiment = mlflow_experiment_name(exp_name)

    # launch new run under the experiment name     
    mlflow.autolog()
    with mlflow.start_run(experiment_id = experiment.experiment_id):
        # hyper parameters
        # solver{‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, default=’lbfgs’
        hyperparams = {'solver': 'saga', 
                        'random_state':0
                      }

        # Training
        logistic = LogisticRegression(**hyperparams)
        logistic.fit(x_train, y_train)

        # score the model
        accuracy=logistic.score(x_test, y_test)
        # print("logistic_score:", logistic_score)
                
        predicted_qualities = logistic.predict(x_test)

        (rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)
                    
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)
        print("  score: %s" % accuracy)
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("score", accuracy)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model      
            mlflow.sklearn.log_model(logistic, exp_name + "-Model", registered_model_name = exp_name + "-Model")
        else:
            mlflow.sklearn.log_model(logistic, exp_name + "-Model")

        logger.info("Model Training completed.\nModel saved in run %s: " % mlflow.active_run().info.run_uuid)          




if __name__ == "__main__":
    warnings.filterwarnings("ignore")    

    # run using following command
    # mlflow server --backend-store-uri sqlite:///mlflow_arem.db  --default-artifact-root ./mlruns

    remote_server_uri = "sqlite:///mlflow_arem.db" #  server URI
    mlflow.set_tracking_uri(remote_server_uri)
    
    mr_uri = mlflow.get_registry_uri()
    logger.info("Current registry uri: {}".format(mr_uri))
    tracking_uri = mlflow.get_tracking_uri()
    logger.info("Current tracking uri: {}".format(tracking_uri))
   
    X_train, X_test, y_train, y_test = data_preprocess()
    
    try:
        LogisticRegression_exp("LogisticRegression",X_train, X_test, y_train, y_test)        
    except Exception as e:
        logger.exception("Error while training: %s", e)
