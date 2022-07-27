import datetime
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics, model_selection, ensemble
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import xgboost as xgb

from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner

@task
def looad_data(path):
    data = pd.read_csv(path)
    data['Loan_Status'] = data['Loan_Status'].replace({'Y' : 1, 'N' : 0}).astype(int)
    data['Property_Area'] = data['Property_Area'].replace({'Rural' : -1, 'Semiurban' : 0, 'Urban':1}).astype(int)
    data['CoapplicantIncome'] = np.log(data['CoapplicantIncome'])
    data['LoanAmount'] = np.log(data['LoanAmount'])
    data['ApplicantIncome'] = np.log(data['ApplicantIncome'])
    data['Loan_Amount_Term'] = np.log(data['Loan_Amount_Term'])
    data.dropna(subset=['CoapplicantIncome', 'ApplicantIncome',  'Loan_Status', 'Loan_Amount_Term', 'LoanAmount'], inplace=True)
    
    return data

@task
def generate_datasets(train_frame):
    num_data = train_frame.select_dtypes(include = [np.number])
    num_data['CoapplicantIncome'] = num_data['CoapplicantIncome'].replace({np.inf : -1, -np.inf: -1})
    num_data['Credit_History'].fillna(0.0, inplace = True)
    
    X = num_data.drop(columns='Loan_Status')
    y = num_data.Loan_Status
    

    return X, y

@task
def train_model(model, X, y):
    numeric_baseline_score = model_selection.cross_val_score(
    model,
    X = X,
    y = y,
    cv = 5,
    scoring = 'accuracy'
    )
    return numeric_baseline_score
  
@task
def estimate_quality(numeric_baseline_score):
    mean = numeric_baseline_score.mean()
    std = numeric_baseline_score.std()
    return mean, std 
    
@flow(task_runner=SequentialTaskRunner())
def nyc_duration_flow():
    train_frame = looad_data('data/loan-train.csv')
    
    X, y = generate_datasets(train_frame).result()
    
    #best model
    best_params = {
        'C': 1.0489085915266627,
        'solver': 'lbfgs'
    }

    best_model = LogisticRegression(**best_params)
    
    numeric_baseline_score = train_model(best_model, X, y)

    score_mean, score_std = estimate_quality(numeric_baseline_score).result()

nyc_duration_flow()

