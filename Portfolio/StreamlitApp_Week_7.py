import os
import sys
import warnings
import tempfile
import tarfile
import posixpath

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib
import boto3
import sagemaker
import shap

from sagemaker.predictor import Predictor
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import NumpyDeserializer
from imblearn.pipeline import Pipeline

warnings.simplefilter("ignore")

# Fix path so Streamlit can find src/
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

if project_root not in sys.path:
    sys.path.append(project_root)

from src.feature_utils import extract_features_pair

# Access Streamlit secrets
aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]


@st.cache_resource
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name="us-east-1"
    )


session = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

# Base features
df_features = extract_features_pair()

MODEL_INFO = {
    "endpoint": aws_endpoint,
    "explainer": "explainer_pair.shap",
    "pipeline": "finalized_pair_model.tar.gz",
    "keys": ["ADBE", "DPZ"],
    "inputs": [
        {"name": k, "type": "number", "min": 0.0, "default": 0.0, "step": 10.0}
        for k in ["ADBE", "DPZ"]
    ],
}


def load_pipeline(_session, bucket, key):
    s3_client = _session.client("s3")
    filename = MODEL_INFO["pipeline"]

    s3_client.download_file(
        Filename=filename,
        Bucket=bucket,
        Key=f"{key}/{os.path.basename(filename)}"
    )

    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        joblib_file = [f for f in tar.getnames() if f.endswith(".joblib")][0]

    return joblib.load(joblib_file)


def load_shap_explainer(_session, bucket, key, local_path):
    s3_client = _session.client("s3")

    if not os.path.exists(local_path):
        s3_client.download_file(
            Filename=local_path,
            Bucket=bucket,
            Key=key
        )

    with open(local_path, "rb") as f:
        return shap.Explainer.load(f)


def call_model_api(input_df):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer()
    )

    try:
        raw_pred = predictor.predict(input_df)
        pred_val = pd.DataFrame(raw_pred).values[-1][0]
        mapping = {-1: "SELL", 0: "HOLD", 1: "BUY"}
        return mapping.get(pred_val, pred_val), 200
    except Exception as e:
        return f"Error: {str(e)}", 500


def display_explanation(input_df, session, aws_bucket):
    explainer_name = MODEL_INFO["explainer"]

    explainer = load_shap_explainer(
        session,
        aws_bucket,
        posixpath.join("explainer", explainer_name),
        os.path.join(tempfile.gettempdir(), explainer_name),
    )

    full_pipeline = load_pipeline(session, aws_bucket, "sklearn-pipeline-deployment")
    preprocessing_pipeline = Pipeline(steps=full_pipeline.steps[:-2])
    input_df_transformed = preprocessing_pipeline.transform(input_df)

    feature_names = full_pipeline[1:4].get_feature_names_out()
    input_df_transformed = pd.DataFrame(input_df_transformed, columns=feature_names)

    shap_values = explainer(input_df_transformed)

    st.subheader("🔍 Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0, :, 0], show=False)
    st.pyplot(fig)

    top_feature = pd.Series(
        shap_values[0, :, 0].values,
        index=shap_values[0, :, 0].feature_names
    ).abs().idxmax()

    st.info(f"**Business Insight:** The most influential factor in this decision was **{top_feature}**.")


st.set_page_config(page_title="ML Deployment", layout="wide")
st.title("👨‍💻 ML Deployment")

with st.form("pred_form"):
    st.subheader("Inputs")
    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp["name"]] = st.number_input(
                inp["name"].replace("_", " ").upper(),
                min_value=inp["min"],
                value=inp["default"],
                step=inp["step"],
            )

    submitted = st.form_submit_button("Run Prediction")

if submitted:
    base_df = df_features.copy()

    # Start from the last real feature row so all required columns stay intact
    new_row = base_df.iloc[[-1]].copy()

    # Overwrite just the ticker input columns with the user-entered values
    for k in MODEL_INFO["keys"]:
        if k in new_row.columns:
            new_row[k] = user_inputs[k]

    input_df = pd.concat([base_df, new_row], ignore_index=True)

    res, status = call_model_api(input_df)

    if status == 200:
        st.metric("Prediction Result", res)
        display_explanation(input_df, session, aws_bucket)
    else:
        st.error(res)



