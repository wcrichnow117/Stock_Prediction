import json
import boto3
import streamlit as st

TARGET_TICKER = 'WYNN'
LABEL_MAP = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}

st.set_page_config(page_title='ML Deployment', layout='centered')
st.title('ML Deployment App')
st.write('Enter a price for the target ticker and send it to your SageMaker endpoint.')

aws_id = st.secrets['aws_credentials']['AWS_ACCESS_KEY_ID']
aws_secret = st.secrets['aws_credentials']['AWS_SECRET_ACCESS_KEY']
aws_token = st.secrets['aws_credentials']['AWS_SESSION_TOKEN']
aws_endpoint = st.secrets['aws_credentials']['AWS_ENDPOINT']


@st.cache_resource
def get_runtime_client():
    session = boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name='us-east-1'
    )
    return session.client('sagemaker-runtime')


runtime_client = get_runtime_client()

with st.form('prediction_form'):
    price_value = st.number_input(
        f'{TARGET_TICKER} price',
        min_value=0.0,
        value=90.0,
        step=1.0,
        help=f'Send a {TARGET_TICKER} price to the endpoint as JSON.'
    )
    submitted = st.form_submit_button('Run Prediction')


if submitted:
    payload = json.dumps({TARGET_TICKER: float(price_value)})

    try:
        response = runtime_client.invoke_endpoint(
            EndpointName=aws_endpoint,
            ContentType='application/json',
            Body=payload
        )

        raw_body = response['Body'].read().decode('utf-8').strip()

        try:
            parsed = json.loads(raw_body)
        except json.JSONDecodeError:
            parsed = raw_body

        pred_value = None
        if isinstance(parsed, list):
            pred_value = parsed[0] if parsed else None
        elif isinstance(parsed, (int, float, str)):
            pred_value = parsed
        elif isinstance(parsed, dict):
            if 'predictions' in parsed and isinstance(parsed['predictions'], list):
                pred_value = parsed['predictions'][0] if parsed['predictions'] else None
            else:
                pred_value = parsed.get('prediction', parsed)

        try:
            pred_value = int(float(pred_value))
        except Exception:
            pass

        display_value = LABEL_MAP.get(pred_value, pred_value)

        st.success('Endpoint call succeeded.')
        st.metric('Prediction', str(display_value))
        st.caption(f'Sent payload: {payload}')
        st.caption(f'Raw endpoint response: {raw_body}')

    except Exception as e:
        st.error(f'Endpoint invocation failed: {e}')
