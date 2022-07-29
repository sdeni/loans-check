import numpy as np
import pandas as pd

import pickle

from sklearn import feature_extraction
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class Model(object):
    def __init__(self) -> None:
        with open('models/best_model.bin', 'rb') as f:
            self.model = pickle.load(f)

        # with open('preprocessing/process_dataframe.bin', 'rb') as g:
        #     pickle.load(g)

    def process_dataframe(self, data):
        if 'Loan_Status' in data.columns:
            data['Loan_Status'] = data['Loan_Status'].replace({'Y': 1, 'N': 0}).astype(int)

        data['Property_Area'] = data['Property_Area'].replace({'Rural': -1, 'Semiurban': 0, 'Urban': 1}).astype(int)
        data['CoapplicantIncome'] = np.log(data['CoapplicantIncome'])
        data['LoanAmount'] = np.log(data['LoanAmount'])
        data['ApplicantIncome'] = np.log(data['ApplicantIncome'])
        data['Loan_Amount_Term'] = np.log(data['Loan_Amount_Term'])

        data.dropna(subset=['CoapplicantIncome', 'ApplicantIncome', 'Loan_Amount_Term', 'LoanAmount'],
                    inplace=True)

        if 'Loan_Status' in data.columns:
            data.dropna(subset=['Loan_Status'], inplace=True)

        num_data = data.select_dtypes(include=[np.number])
        num_data['Credit_History'].fillna(0.0, inplace=True)
        num_data['CoapplicantIncome'] = num_data['CoapplicantIncome'].replace({np.inf: -1, -np.inf: -1})
        return num_data

    def test(self):
        data = pd.read_csv('data/loan-train.csv')
        num_data = self.process_dataframe(data)
        print(num_data.isnull().values.any())
        X = num_data.drop(columns='Loan_Status')
        y_test = num_data.Loan_Status

        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y_pred, y_test)
        print('accuracy on train data: ', accuracy)

    def predict(self, app_form_data):
        df = pd.DataFrame.from_dict(app_form_data)
        num_data = self.process_dataframe(df)
        return self.model.predict(num_data)
