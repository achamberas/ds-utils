import os
import pandas as pd
# import pandas_gbq as pdgbq
import snowflake.connector

import google.auth
from google.oauth2 import service_account
from google.cloud import bigquery

SF_USER = os.getenv("SF_USER", 'READONLY')
SF_PASSWORD = os.getenv("SF_PASSWORD", '')
SF_ACCOUNT = os.getenv("SF_ACCOUNT", '')
SF_WAREHOUSE = os.getenv("SF_WAREHOUSE", '')
SF_DATABASE = os.getenv("SF_DATABASE", '')
SF_SCHEMA = os.getenv("SF_SCHEMA", '')

GOOGLE_PROJECT = os.getenv("GOOGLE_PROJECT", 'gristmill5')
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", 'creds/gristmill5-e521e2f08f35.json')


def sf_conn(sql):

    try:
        conn = snowflake.connector.connect(
            user=SF_USER,
            password=SF_PASSWORD,
            account=SF_ACCOUNT,
            warehouse=SF_WAREHOUSE,
            database=SF_DATABASE,
            schema=SF_SCHEMA
        )

        cur = conn.cursor()
        cur.execute(sql)
        df = cur.fetch_pandas_all()

        return df

    except Exception as e:
        print(e)
        return 'error running query'


def bq_conn(sql):

    try:
        credentials = service_account.Credentials.from_service_account_file(GOOGLE_APPLICATION_CREDENTIALS)
        client = bigquery.Client(GOOGLE_PROJECT, credentials)
        df = client.query(sql, project=GOOGLE_PROJECT).to_dataframe()

        return df

    except Exception as e:
        print(e)
        return 'error running query'
