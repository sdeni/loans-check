import datetime
import matplotlib
import numpy as np
import pandas as pd
from sklearn import metrics, model_selection, ensemble
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner

from pandas_profiling import ProfileReport

@task
def load_data(path):
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

    return train_test_split(X,y, test_size=0.2, random_state=0)

@task
def train_model(X, y):

    #best model
    best_params = {
        'C': 1.0489085915266627,
        'solver': 'lbfgs'
    }

    best_model = LogisticRegression(**best_params)
    
    best_model.fit(X, y)
    
    return best_model

@task
def estimate_quality(model, X, y):

    numeric_baseline_score = model_selection.cross_val_score(
    model,
    X = X,
    y = y,
    cv = 5,
    scoring = 'accuracy'
    )
    
    mean = numeric_baseline_score.mean()
    std = numeric_baseline_score.std()
    
    return mean, std 

def output_accuracy(accuracy):
    f = open("model_quality.txt", "w")
    f.write("accuracy on test data:" + str(accuracy))
    f.close()

@flow(task_runner=SequentialTaskRunner())
def loan_check_flow():
    train_frame = load_data('data/loan-train.csv')
    
    X_train, X_test, y_train, y_test = generate_datasets(train_frame).result()
    
    model = train_model(X_train, y_train)

    score_mean, score_std = estimate_quality(model,X_test, y_test).result()

    return X_train, score_mean


data, accuracy = loan_check_flow().result()

output_accuracy(accuracy)

profile = ProfileReport(data)
profile.to_file("Data_Quality.html")

