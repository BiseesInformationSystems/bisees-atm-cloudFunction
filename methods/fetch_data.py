from google.cloud import bigquery
import pandas as pd
import datetime
from sqlalchemy import text 
from sqlalchemy import create_engine 
from google.cloud import aiplatform
from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_file('service_account.json')
db = create_engine('bigquery://', credentials_path='service_account.json')

def fetch_data_df(query): 
    df = pd.read_sql(query, con=db)
    return df


def predict_tabular_regression_sample(
    project,
    location,
    endpoint_name,
    instances,
):
    aiplatform.init(project=project, location=location, credentials=credentials)

    endpoint = aiplatform.Endpoint.list(f"display_name={endpoint_name}")[0]
    instances = instances.values.tolist()
    response = endpoint.predict(instances=instances)

    #for prediction_ in response.predictions:
    #    print(prediction_)
    return response.predictions


def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])