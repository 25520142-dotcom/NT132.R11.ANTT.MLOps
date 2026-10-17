#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import pickle
import xgboost as xgb

import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment")


# In[6]:


from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
from pathlib import Path
models_folder = Path('models').mkdir(exist_ok=True)


# In[ ]:


def read_dataframe(year, month):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    Categorical = ['PULocationID', 'DOLocationID']
    df[Categorical] = df[Categorical].astype(str)
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    return df


# In[ ]:


df_train = read_dataframe(year = 2021, month = 1)
df_val = read_dataframe(year = 2021, month=2)


# In[ ]:


def feature_engineering(df, dv = None):
    Categorical =  ['PU_DO'] #['PULocationID', 'DOLocationID']
    Numerical = ['trip_distance']
    dicts = df[Categorical + Numerical].to_dict(orient = 'records')

    if dv is None:
        dv = DictVectorizer()
        x = dv.fit_transform(dicts)
    else:
        x = dv.transform(dicts)

    return x, dv


# In[ ]:


def train_model(x_train, y_train, x_val, y_val, dv):
    with mlflow.start_run() as run:
        train = xgb.DMatrix(x_train, y_train)
        valid = xgb.DMatrix(x_val, y_val)

        best_params = {
            'learning_rate': 0.0958,
            'max_depth': 30,
            'min_child_weight': 1.06059,
            'objective': 'reg:linear',
            'reg_alpha': 0.018,
            'reg_lambda': 0.01165,
            'seed': 42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params = best_params,
            dtrain = train,
            num_boost_round = 30,
            evals = [(valid, 'validation')],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)

        mlflow.log_metric("rmse", rmse)

        with open("models/preprocessor.b", "wb") as f:
            pickle.dump(dv, f)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

        return run.info.run_id


# In[ ]:


def run(year, month):
    #Downloading data
    df_train = read_dataframe(year = year, month = month)
    next_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1
    df_val = read_dataframe(year = next_year, month = next_month)

    #Feature Engineer
    x_train, dv = feature_engineering(df_train)
    x_val, dv = feature_engineering(df_val, dv)

    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    #Skipped Paratermeter tuning because already done

    #Train and log
    run_id = train(x_train, y_train, x_val, y_val, dv)
    return run_id


# In[ ]:


if __name == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train model to predict taxi duration')
    parser.add_argument('--year', type = int, required=True, help='Year of the data')
    parser.add_argument('--month', type = int, required=True, help='Month of the data')

    run(year=args.year, month=args.month)

