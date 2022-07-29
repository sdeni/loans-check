from flask import Flask, render_template, request

from ml import Model

app = Flask(__name__)

@app.route("/check_eligibility", methods=['POST'])
def check_eligibility():
    data = request.form
    model = Model()

    try:
        data_item = {
            "Loan_ID": ["realtime"],
            "Gender": [data["Gender"]],
            "Married": [data["Married"] if "Married" in data else "No"],
            "Dependents": [data["Dependents"] if "Dependents" in data else "0"],
            "Education": data["Education"] if "Education" in data else "Not graduated",
            "Self_Employed": data["Self_Employed"] if "Self_Employed" in data else "No",
            "ApplicantIncome": int(data["ApplicantIncome"]) if "ApplicantIncome" in data and data["ApplicantIncome"] else 0,
            "CoapplicantIncome": float(data["CoapplicantIncome"]) if "CoapplicantIncome" in data and float(data["CoapplicantIncome"]) else 0.0,
            "LoanAmount": float(data["LoanAmount"]) if "LoanAmount" in data and data["LoanAmount"] else 1.0,
            "Loan_Amount_Term": int(data["Loan_Amount_Term"]) if "Loan_Amount_Term" in data and data["Loan_Amount_Term"] else 360,
            "Credit_History":  float(data["Credit_History"]) if "Credit_History" in data and data["Credit_History"] else 1.0,
            "Property_Area":  (data["Property_Area"]) if "Property_Area" in data and data["Property_Area"] else 'Urban'
        }

    # data_item = {"ApplicantIncome": 12333.0, "CoapplicantIncome": 0, "Credit_History": 1.0, "Dependents": [0], "LoanAmount": 5000.0,
    #  "Loan_Amount_Term": 180, "Loan_ID": ["realtime"], "Property_Area": "Urban"}

#        return data_item

        data_item["Gender"] = ["Male"]
        data_item["Married"] = "Yes"
        # data_item["Dependents"] = "0"
        # data_item["Education"] = "Graduate"
        # data_item["Self_Employed"] = "No"
        # data_item["ApplicantIncome"] = 5849
        # data_item["CoapplicantIncome"] = 0.0
        # data_item["LoanAmount"] = 128
        # data_item["Loan_Amount_Term"] = 360
        # data_item["Credit_History"] = 1.0
        # data_item["Property_Area"] = "Urban"

        # OK
        # {"ApplicantIncome": 5849, "CoapplicantIncome": 0.0, "Credit_History": 1.0, "Dependents": "0",
        #  "Education": "Graduate", "Gender": ["Male"], "LoanAmount": 128, "Loan_Amount_Term": 360,
        #  "Loan_ID": ["realtime"], "Married": "Yes", "Property_Area": "Urban", "Self_Employed": "No"}

        # {
        #     "ApplicantIncome": 2.0,
        #     "CoapplicantIncome": 0,
        #     "Credit_History": 1.0,
        #     "Dependents": [
        #         0
        #     ],
        #     "Education": "1",
        #     "Gender": [
        #         "Female"
        #     ],
        #     "LoanAmount": 324.0,
        #     "Loan_Amount_Term": 360,
        #     "Loan_ID": [
        #         "realtime"
        #     ],
        #     "Married": [
        #         "No"
        #     ],
        #     "Property_Area": "Urban",
        #     "Self_Employed": "No"
        # }

        data_item2 = {
            "Loan_ID": ["realtime"],
            "Gender": ["Male"],
            "Married": "Yes",
            "Dependents": "0",
            "Education": "Graduate",
            "Self_Employed": "No",
            "ApplicantIncome": 5849,
            "CoapplicantIncome": 0.0,
            "LoanAmount": 128,
            "Loan_Amount_Term": 360,
            "Credit_History": 1.0,
            "Property_Area": "Urban"
        }

        res = model.predict(data_item)
        return render_template('result.html', result=res)
    except:
        return render_template('index.html', error=True, data=data_item)


@app.route("/")
@app.route('/index')
def index():
#    model = Model()
#    model.test()

    # data_item = {
    #     "Loan_ID": ["realtime"],
    #     "Gender": ["Male"],
    #     "Married": "Yes",
    #     "Dependents": "0",
    #     "Education": "Graduate",
    #     "Self_Employed": "No",
    #     "ApplicantIncome": 5849,
    #     "CoapplicantIncome": 0.0,
    #     "LoanAmount": 128,
    #     "Loan_Amount_Term": 360,
    #     "Credit_History": 1.0,
    #     "Property_Area": "Urban"
    # }
    #
    # res = model.predict(data_item)

    return render_template('index.html')
