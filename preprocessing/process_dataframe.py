import numpy as np

def process_dataframe(data):
        data['Loan_Status'] = data['Loan_Status'].replace({'Y': 1, 'N': 0}).astype(int)
        data['Property_Area'] = data['Property_Area'].replace({'Rural': -1, 'Semiurban': 0, 'Urban': 1}).astype(int)
        data['CoapplicantIncome'] = np.log(data['CoapplicantIncome'])
        data['LoanAmount'] = np.log(data['LoanAmount'])
        data['ApplicantIncome'] = np.log(data['ApplicantIncome'])
        data['Loan_Amount_Term'] = np.log(data['Loan_Amount_Term'])
        data.dropna(subset=['CoapplicantIncome', 'ApplicantIncome', 'Loan_Status', 'Loan_Amount_Term', 'LoanAmount'],
                    inplace=True)
        num_data = data.select_dtypes(include=[np.number])
        num_data['Credit_History'].fillna(0.0, inplace=True)
        num_data['CoapplicantIncome'] = num_data['CoapplicantIncome'].replace({np.inf: -1, -np.inf: -1})
        return num_data
