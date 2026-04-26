import json
import warnings

import boto3
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

warnings.simplefilter("ignore")

st.set_page_config(page_title="LendingClub Default Predictor", layout="wide")

st.title("LendingClub Loan Default Predictor")
st.write(
    "Enter borrower and loan information to estimate whether the loan is higher or lower default risk."
)

def get_secret_value(key, default=None):
    if "aws_credentials" in st.secrets and key in st.secrets["aws_credentials"]:
        return st.secrets["aws_credentials"][key]
    return st.secrets.get(key, default)

AWS_ACCESS_KEY_ID = get_secret_value("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = get_secret_value("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = get_secret_value("AWS_SESSION_TOKEN")
AWS_ENDPOINT = get_secret_value("AWS_ENDPOINT")

if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY or not AWS_SESSION_TOKEN or not AWS_ENDPOINT:
    st.error(
        "Missing AWS credentials or endpoint. Add AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, "
        "AWS_SESSION_TOKEN, and AWS_ENDPOINT in Streamlit Advanced Settings / Secrets."
    )
    st.stop()

@st.cache_resource
def get_sagemaker_runtime_client(access_key, secret_key, session_token):
    return boto3.client(
        "sagemaker-runtime",
        region_name="us-east-1",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        aws_session_token=session_token,
    )

runtime_client = get_sagemaker_runtime_client(
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_SESSION_TOKEN,
)

COMMON_FEATURE_COLUMNS = [
    "loan_amnt",
    "funded_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "dti",
    "delinq_2yrs",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "application_type",
    "emp_length_num",
    "term_months",
    "issue_year",
    "issue_month",
    "credit_history_years",
    "loan_to_income",
    "installment_to_income",
    "revol_bal_to_income",
    "fico_avg",
    "credit_util_x_dti",
    "interest_x_term",
    "income_log",
    "loan_amount_log",
    "total_acc_log",
    "income_per_open_acc",
]

def safe_log1p(value):
    import math
    return math.log1p(max(float(value), 0.0))

def build_feature_row(values):
    loan_amnt = float(values["loan_amnt"])
    funded_amnt = float(values["funded_amnt"])
    int_rate = float(values["int_rate"])
    installment = float(values["installment"])
    annual_inc = float(values["annual_inc"])
    dti = float(values["dti"])
    revol_bal = float(values["revol_bal"])
    revol_util = float(values["revol_util"])
    fico_low = float(values["fico_range_low"])
    fico_high = float(values["fico_range_high"])
    total_acc = float(values["total_acc"])
    open_acc = float(values["open_acc"])
    term_months = 36 if values["term"] == "36 months" else 60

    emp_map = {
        "< 1 year": 0,
        "1 year": 1,
        "2 years": 2,
        "3 years": 3,
        "4 years": 4,
        "5 years": 5,
        "6 years": 6,
        "7 years": 7,
        "8 years": 8,
        "9 years": 9,
        "10+ years": 10,
    }
    emp_length_num = emp_map.get(values["emp_length"], 0)

    row = {
        "loan_amnt": loan_amnt,
        "funded_amnt": funded_amnt,
        "term": values["term"],
        "int_rate": int_rate,
        "installment": installment,
        "grade": values["grade"],
        "sub_grade": values["sub_grade"],
        "emp_length": values["emp_length"],
        "home_ownership": values["home_ownership"],
        "annual_inc": annual_inc,
        "verification_status": values["verification_status"],
        "purpose": values["purpose"],
        "dti": dti,
        "delinq_2yrs": float(values["delinq_2yrs"]),
        "fico_range_low": fico_low,
        "fico_range_high": fico_high,
        "inq_last_6mths": float(values["inq_last_6mths"]),
        "open_acc": open_acc,
        "pub_rec": float(values["pub_rec"]),
        "revol_bal": revol_bal,
        "revol_util": revol_util,
        "total_acc": total_acc,
        "application_type": values["application_type"],
        "emp_length_num": emp_length_num,
        "term_months": term_months,
        "issue_year": int(values["issue_year"]),
        "issue_month": int(values["issue_month"]),
        "credit_history_years": float(values["credit_history_years"]),
        "loan_to_income": loan_amnt / (annual_inc + 1),
        "installment_to_income": installment / (annual_inc + 1),
        "revol_bal_to_income": revol_bal / (annual_inc + 1),
        "fico_avg": (fico_low + fico_high) / 2,
        "credit_util_x_dti": revol_util * dti,
        "interest_x_term": int_rate * term_months,
        "income_log": safe_log1p(annual_inc),
        "loan_amount_log": safe_log1p(loan_amnt),
        "total_acc_log": safe_log1p(total_acc),
        "income_per_open_acc": annual_inc / (open_acc + 1),
    }

    return {col: row.get(col) for col in COMMON_FEATURE_COLUMNS}

def call_endpoint(input_row):
    payload = json.dumps([input_row])

    response = runtime_client.invoke_endpoint(
        EndpointName=AWS_ENDPOINT,
        ContentType="application/json",
        Body=payload,
    )

    result = json.loads(response["Body"].read().decode("utf-8"))

    if isinstance(result, list) and len(result) > 0:
        return result[0]
    return result

with st.form("loan_prediction_form"):
    st.subheader("Loan and Borrower Inputs")

    col1, col2, col3 = st.columns(3)

    with col1:
        loan_amnt = st.number_input("Loan Amount", min_value=500.0, max_value=50000.0, value=10000.0, step=500.0)
        funded_amnt = st.number_input("Funded Amount", min_value=500.0, max_value=50000.0, value=10000.0, step=500.0)
        term = st.selectbox("Loan Term", ["36 months", "60 months"])
        int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=40.0, value=13.0, step=0.25)
        installment = st.number_input("Monthly Installment", min_value=0.0, max_value=2500.0, value=330.0, step=10.0)
        purpose = st.selectbox(
            "Loan Purpose",
            ["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "small_business", "other"],
        )

    with col2:
        annual_inc = st.number_input("Annual Income", min_value=0.0, max_value=1000000.0, value=65000.0, step=1000.0)
        dti = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=100.0, value=18.0, step=0.5)
        fico_range_low = st.number_input("FICO Range Low", min_value=300.0, max_value=850.0, value=680.0, step=1.0)
        fico_range_high = st.number_input("FICO Range High", min_value=300.0, max_value=850.0, value=684.0, step=1.0)
        grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"], index=2)
        sub_grade = st.selectbox(
            "Loan Subgrade",
            [
                "A1", "A2", "A3", "A4", "A5",
                "B1", "B2", "B3", "B4", "B5",
                "C1", "C2", "C3", "C4", "C5",
                "D1", "D2", "D3", "D4", "D5",
                "E1", "E2", "E3", "E4", "E5",
                "F1", "F2", "F3", "F4", "F5",
                "G1", "G2", "G3", "G4", "G5",
            ],
            index=11,
        )

    with col3:
        emp_length = st.selectbox(
            "Employment Length",
            ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"],
            index=5,
        )
        home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"], index=1)
        verification_status = st.selectbox("Verification Status", ["Verified", "Source Verified", "Not Verified"], index=1)
        application_type = st.selectbox("Application Type", ["Individual", "Joint App"])
        delinq_2yrs = st.number_input("Delinquencies in Past 2 Years", min_value=0.0, max_value=20.0, value=0.0, step=1.0)
        inq_last_6mths = st.number_input("Inquiries in Last 6 Months", min_value=0.0, max_value=20.0, value=1.0, step=1.0)

    st.subheader("Credit Account Details")
    col4, col5, col6 = st.columns(3)

    with col4:
        open_acc = st.number_input("Open Credit Accounts", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
        total_acc = st.number_input("Total Credit Accounts", min_value=0.0, max_value=200.0, value=25.0, step=1.0)

    with col5:
        pub_rec = st.number_input("Public Records", min_value=0.0, max_value=20.0, value=0.0, step=1.0)
        revol_bal = st.number_input("Revolving Balance", min_value=0.0, max_value=500000.0, value=12000.0, step=500.0)

    with col6:
        revol_util = st.number_input("Revolving Utilization (%)", min_value=0.0, max_value=150.0, value=45.0, step=1.0)
        credit_history_years = st.number_input("Credit History Years", min_value=0.0, max_value=80.0, value=15.0, step=1.0)

    with st.expander("Optional issue date inputs"):
        issue_year = st.number_input("Issue Year", min_value=2007, max_value=2026, value=2016, step=1)
        issue_month = st.number_input("Issue Month", min_value=1, max_value=12, value=6, step=1)

    submitted = st.form_submit_button("Predict Default Risk")

user_values = {
    "loan_amnt": loan_amnt,
    "funded_amnt": funded_amnt,
    "term": term,
    "int_rate": int_rate,
    "installment": installment,
    "grade": grade,
    "sub_grade": sub_grade,
    "emp_length": emp_length,
    "home_ownership": home_ownership,
    "annual_inc": annual_inc,
    "verification_status": verification_status,
    "purpose": purpose,
    "dti": dti,
    "delinq_2yrs": delinq_2yrs,
    "fico_range_low": fico_range_low,
    "fico_range_high": fico_range_high,
    "inq_last_6mths": inq_last_6mths,
    "open_acc": open_acc,
    "pub_rec": pub_rec,
    "revol_bal": revol_bal,
    "revol_util": revol_util,
    "total_acc": total_acc,
    "application_type": application_type,
    "credit_history_years": credit_history_years,
    "issue_year": issue_year,
    "issue_month": issue_month,
}

input_row = build_feature_row(user_values)
input_df = pd.DataFrame([input_row])

with st.expander("Preview model input sent to SageMaker"):
    st.dataframe(input_df)

if submitted:
    try:
        result = call_endpoint(input_row)

        prediction = result.get("prediction")
        probability = result.get("default_probability")

        st.subheader("Prediction Result")

        if probability is not None:
            st.metric("Predicted Default Probability", f"{float(probability):.2%}")
        else:
            st.write(result)

        if prediction == 1:
            st.warning("The model classifies this loan as higher default risk.")
        elif prediction == 0:
            st.success("The model classifies this loan as lower default risk.")

        st.subheader("Decision Transparency")

        shap_features = result.get("shap_features", [])
        shap_values = result.get("shap_values", [])

        if shap_features and shap_values:
            shap_df = pd.DataFrame(
                {"Feature": shap_features, "SHAP Value": shap_values}
            ).sort_values("SHAP Value")

            fig, ax = plt.subplots(figsize=(9, 5))
            ax.barh(shap_df["Feature"], shap_df["SHAP Value"])
            ax.set_xlabel("SHAP Value")
            ax.set_title("Top Factors Affecting This Prediction")
            st.pyplot(fig)

            top_feature = shap_df.iloc[shap_df["SHAP Value"].abs().argmax()]["Feature"]
            st.info(f"The most influential factor in this prediction was **{top_feature}**.")
        else:
            st.info(
                "The prediction worked, but SHAP values were not returned by the endpoint. "
                "Use the notebook SHAP plots as backup evidence for the rubric."
            )
            if "shap_error" in result:
                st.caption(result["shap_error"])

    except Exception as e:
        st.error("Prediction failed.")
        st.write(e)
        st.info(
            "Double-check that your AWS credentials are current, the endpoint is InService, "
            "and AWS_ENDPOINT in Streamlit secrets exactly matches the SageMaker endpoint name."
        )
