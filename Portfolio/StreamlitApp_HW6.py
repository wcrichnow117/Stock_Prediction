import warnings
import numpy as np
import pandas as pd
import streamlit as st
import boto3
import sagemaker

from sagemaker.predictor import Predictor
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import JSONDeserializer

warnings.simplefilter("ignore")

st.set_page_config(page_title="ADBE Stock Return Predictor", layout="wide")

st.title("ADBE Stock Return Predictor")
st.write("Enter model inputs below to predict Adobe's next-day stock return.")

aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

@st.cache_resource
def get_session():
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name="us-east-1"
    )

session = get_session()
sagemaker_session = sagemaker.Session(boto_session=session)

predictor = Predictor(
    endpoint_name=aws_endpoint,
    sagemaker_session=sagemaker_session,
    serializer=NumpySerializer(),
    deserializer=JSONDeserializer()
)

with st.form("prediction_form"):
    st.subheader("Model Inputs")

    adbe = st.number_input("ADBE Price", value=500.00)
    sentiment = st.number_input("Sentiment Score", value=0.00, min_value=-1.0, max_value=1.0)
    return_1d = st.number_input("1-Day Return", value=0.00)
    volatility_5d = st.number_input("5-Day Volatility", value=0.01)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([{
        "ADBE": adbe,
        "sentiment_textblob": sentiment,
        "ADBE_return_1d": return_1d,
        "ADBE_volatility_5d": volatility_5d
    }])

    try:
        prediction = predictor.predict(input_df.values)
        pred_value = float(np.array(prediction).flatten()[0])

        st.metric("Predicted Next-Day Return", round(pred_value, 5))

        if pred_value > 0:
            st.success("Model signal: Positive expected return")
        elif pred_value < 0:
            st.error("Model signal: Negative expected return")
        else:
            st.info("Model signal: Neutral")

    except Exception as e:
        st.error(f"Prediction failed: {e}")



