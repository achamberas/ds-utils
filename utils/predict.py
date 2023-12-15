import os
import pandas as pd
import numpy as np
import dill
import re
import scipy
# import sksurv
import json

from sklearn import preprocessing
# from utils.feature_engineering import *
# from connectors.cloud_storage.cloud_storage import cloud_storage_read
from datetime import datetime, timedelta

GOOGLE_BUCKET = os.getenv("GOOGLE_BUCKET", False)
GOOGLE_BUCKET = "tesla-gristmill5"

def load_model(model, model_cache='USE'):
    # get model details from picklefile
    if not os.path.isfile('models/' + model + '.pkl') or model_cache == 'CLEAR':
        cloud_storage_read(GOOGLE_BUCKET, 'models/' + model +'.pkl')

    # modelfile = open('/tmp/models-' + model + '.pkl','rb')
    modelfile = open('models/' + model + '.pkl','rb')
    
    results = dill.load(modelfile)
    modelfile.close() 
    
    return results

def predict(data, model_pkg, israndom, scenario={}, model_cache='USE'):

    if len(data) == 0:
        data['prediction'] = None
        return data
    
    # load model if stored remotely
    if model_pkg.__class__ == str:
        model_pkg = load_model(model_pkg, model_cache='USE')
    
    model = model_pkg['model']
    model_type = model_pkg['model_type']
    stdev = None if 'stdev' not in model_pkg else model_pkg['stdev']
    features = model_pkg['features']
    columns = [f['name'] for f in features]
    
    # check for missing features in dataset
    x_cols = {'vc':data.columns}
    c = pd.DataFrame.from_dict(x_cols)
    f = pd.DataFrame.from_dict(features).reset_index(drop=True)

    if len(f) > 0:
        r = f.merge(c, how='left', left_on='name', right_on='vc')[['name', 'vc']]
        m = r['name'][r['vc'].isna()]
    
        if len(m) > 0:
            print('Missing features:')
            print(m)
            # return
    
    # X = pd.DataFrame(index=range(0, len(data)))
    ip = data.copy()
    X = pd.DataFrame(index=ip.index)
    
    # apply feature transformations
    for f in features:
        fname = f['name']
        
        for key in data.columns:
            if (re.match(key, fname) and data[key].dtypes == 'object') or key == fname:
                # transform date feature to dateparts
                if f['transform'] == 'dtimes':
                    test_date = pd.to_datetime(data[key], errors='coerce', infer_datetime_format=True)
                    if test_date.notna().any() \
                        and (re.match(r"datetime64(.*)", data[key].dtypes.name) \
                        or data[key].dtypes == 'object'):
                        data[key] = pd.to_datetime(data[key], errors='coerce', infer_datetime_format=True)
                        X[key] = data[key]
                        add_datepart(X, key, drop=True, prefix=key+'_')
                    
                # check if column is a string and generate dummy/ohe vars
                elif data[key].dtypes.name in ['object','category']:
                    for key_value in f['values']:
                        X = X.copy()
                        X[str(fname)+'_'+str(key_value)] = np.where(data[key].eq(key_value), 1, 0)
                        
                # standardize numeric if model does so
                elif data[key].dtypes in ['float64', 'int64', 'int8'] and f['transform'] == 'stdize':
                    min_x = f['min']
                    max_x = f['max']
                    X[key] = (data[[key]] - min_x) / max_x
                    #min_max_scaler = preprocessing.MinMaxScaler()
                    #X[key] = min_max_scaler.fit_transform(data[[key]])
                else:
                    X[key] = data[key]

    # run predictions
    X = X.fillna(0)

    if model_type == 'survival':

        period = results['period']
        surv = model.predict_survival_function(X, return_array=False)
        data['index'] = data.index
        
        # find max possible period
        data[period + '_max'] = np.asarray(model.event_times_).max()
        data[period + '_adj'] = data[[period, period + '_max']].min(axis=1).astype(int)
        
        data.loc[:,'prediction'] = data[[period + '_adj', 'index']].apply(lambda x: surv[x[1]](x[0]), axis=1)
        data = data.drop(columns=['index', period + '_max', period + '_adj'], axis=1)

        if israndom:
            data['randomuniform'] = np.random.uniform(0, 1, len(data))
            data['prediction'] = data[['prediction', 'randomuniform']].apply(lambda x: 1 if x[1] > x[0] else 0, axis=1)
            data = data.drop(columns=['randomuniform'], axis=1)

    elif model_type == 'classification':
        data.loc[:,'prediction'] = model.predict(X)
        if israndom:
            data['randomuniform'] = np.random.uniform(0, 1, len(data))
            data['prediction'] = data[['prediction', 'randomuniform']].apply(lambda x: 1 if x[1] > x[0] else 0, axis=1)
            data = data.drop(columns=['randomuniform'], axis=1)

    elif model_type == 'regression':
        data.loc[:,'prediction'] = model.predict(X)
        if israndom:
            data['prediction'] = data['prediction'] * (1 + np.random.normal(0, stdev, len(data)))

    elif model_type == 'simulation':
        model = dill.loads(model)
        m = model()
        data = m.simulation(scenario, data)
        
    elif model_type == '3pm api':
        model = dill.loads(model)
        m = model()
        data = m.score(data, features)
        
    return data
