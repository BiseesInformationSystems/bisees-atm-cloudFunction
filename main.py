from datetime import datetime, timedelta
import functions_framework
import numpy as np 
import math
import pandas as pd
from random import randint
import math
from random import randint
from methods import fetch_data, add_features


def roundup(x):
    return int(math.ceil(x / 20.0)) * 20

def get_prediction(atm_id, forcasting_window): 
    pred_x = add_features.add_features(atm_id, int(forcasting_window))
    pred = fetch_data.predict_tabular_regression_sample(
        project="exepnoproject",
        endpoint_name='atms_prediction_endpoint',
        location="europe-west1",
        instances=pred_x
    )
    pred = np.expm1(pred)
    # print(pred)
    pred_x['prediction'] = pred * 1.7
    pred_x['prediction'] = pred_x['prediction'].map(lambda x: math.ceil(x))
    pred_x = pred_x.filter(['date', 'prediction'], axis=1).reset_index()
    pred_x['prediction'] = pred_x['prediction'].map(lambda x: roundup(x))

    return pred_x.to_dict('records')

def get_test_performance(atm_id): 
    test = fetch_data.fetch_data_df(f'''
        SELECT 
            date, 
            CAST(disbursement_amnt AS NUMERIC) as withdr_amnt, 
            ROUND(CAST(disbursement_amnt_pred AS NUMERIC)*1.7, 0) as withdr_amnt_pred, 
            mse, 
            rmse, 
            mape 
        FROM `atms_ai_models.xgboost_test_results`
        WHERE atmId='{atm_id}'
        ORDER BY date ASC;  
    ''')

    test['withdr_amnt_pred'] = test['withdr_amnt_pred'].map(lambda x: roundup(x))

    testPerf = {
        'metadata': {
            'mse': float(test['mse'].values[0]),
            'rmse': float(test['rmse'].values[0]),
            'mape': float(test['mape'].values[0])
        }, 
        'data': test.filter(['date', 'withdr_amnt', 'withdr_amnt_pred'], axis=1).to_dict('records')
    }

    return testPerf

def get_top_atms_by_rmse(n=10): 
    top_atms = fetch_data.fetch_data_df(f'''
        SELECT DISTINCT 
            atmId, 
            CAST(rmse AS NUMERIC) AS rmse
        FROM `atms_ai_models.xgboost_test_results`
        ORDER BY rmse ASC 
        LIMIT {n}; 
    ''')

    return top_atms.to_dict('records')


def get_feature_importances(): 
    features = fetch_data.fetch_data_df(f'''
        SELECT 
            feature, 
            CAST(Value AS NUMERIC) value
        FROM `atms_ai_models.xgboost_feature_importance` 
        ORDER BY Value DESC;
    ''')
    features = features[features['feature']!='atmId']
    return features.to_dict('records')


def get_daily_withr_amnt(start, end, code): 
    historical_data = fetch_data.fetch_data_df(f'''
        SELECT 
            date, 
            code, 
            atmId,
            CAST(disbursement_amnt AS NUMERIC) AS withdrAmnt
        FROM  `atms_ai_models.atm_trans_OffSite_tmp2` 
        WHERE date>='{start}' AND date<='{end}' AND atmId = {code}
        ORDER BY date, atmId;
    ''')
    
    return historical_data.to_dict('records')
    # print(historical_data.head())


def get_total_withdr_amnt(start, end): 
    total_daily = fetch_data.fetch_data_df(f'''
        SELECT 
            date, 
            SUM(CAST(disbursement_amnt AS NUMERIC)) AS withdrAmnt
        FROM  `atms_ai_models.atm_trans_OffSite_tmp2` 
        WHERE date>='{start}' AND date<='{end}'
        GROUP BY date
        ORDER BY date;
    ''')
    total_daily['SMA1W'] = round(total_daily['withdrAmnt'].rolling(7).mean().fillna(0), 0)
    total_daily['EMA1W'] = round(total_daily['withdrAmnt'].ewm(span=7, adjust=False).mean().fillna(0), 0)
    return total_daily.to_dict('records')

def get_yearly_withdr_amnt(atmId): 
    total_yearly = fetch_data.fetch_data_df(f'''
        SELECT 
            FORMAT_DATE('%Y', date) AS date, 
            SUM(CAST(disbursement_amnt AS NUMERIC)) AS withdrAmnt
        FROM  `atms_ai_models.atm_trans_OffSite_tmp2` 
        WHERE atmId={atmId}
        GROUP BY date
        ORDER BY date;
    ''')

    return total_yearly.to_dict('records')

def get_withr_amnt(atmId): 
    atm_total = fetch_data.fetch_data_df(f'''
        SELECT  
            code,
            SUM(CAST(disbursement_amnt AS NUMERIC)) AS withdrAmnt
        FROM  `atms_ai_models.atm_trans_OffSite_tmp2` 
        WHERE Date >= DATE_ADD(CURRENT_DATE(), INTERVAL -365 DAY) AND atmId={atmId}
        GROUP BY code;
    ''')
    return atm_total.to_dict('records')

def get_top_atms_by_withr_amnt(atmId): 
    top_atms = fetch_data.fetch_data_df(f'''
        SELECT  
            code,
            SUM(CAST(disbursement_amnt AS NUMERIC)) AS withdrAmnt
        FROM  `atms_ai_models.atm_trans_OffSite_tmp2` 
        WHERE Date >= DATE_ADD(CURRENT_DATE(), INTERVAL -365 DAY)
        GROUP BY code
        ORDER BY withdrAmnt DESC LIMIT 5;
    ''')
    top_atms = top_atms.append(get_withr_amnt(atmId), ignore_index=True)
    top_atms = top_atms.drop_duplicates()
    return top_atms.to_dict('records')


def get_ha_metadata(codes): 
    start = (datetime.today() - timedelta(days=365*2)).date()
    end = datetime.today().date()
    data = pd.DataFrame(get_daily_withr_amnt(start, end, codes))

    windows = [7, 30, 90, 365]
    lastest_date = data['date'].max()

    res = {}
    
    for window in windows: 
        wind_data = data[data['date']>=(lastest_date-timedelta(days=window))]
        total_amount = round(wind_data['withdrAmnt'].sum(), 2)

        wind_data = data[
            (data['date']>=(lastest_date-timedelta(days=window*2))) &
            (data['date']<=(lastest_date-timedelta(days=window)))
        ]
        old_total_amount = round(wind_data['withdrAmnt'].sum(), 2)
        pct_increase = round((total_amount - old_total_amount) / old_total_amount * 100, 1)
        res[f'{window}_day'] = {
            'withdrAmnt': total_amount, 
            'pctIncrease': pct_increase
        }

    return res


def get_remaining(atm_df, atm_capacity): 
    remaining = atm_df.sort_values(by=['date'], ascending=False).reset_index(drop=True)
    total = 0
    for i, row in remaining.iterrows(): 
        if row['money_fulfilled'] == 0: 
            total += row['disbursement_amnt']
        else: 
            # total += row['disbursement_amnt']
            break

    return atm_capacity - total


def fake_dates():
    dates = [str((datetime.now() + timedelta(d)).date()) for d in range(30)]
    l = []
    step = randint(3, 6)
    # indexes=[]
    while True: 
        try: 
            l.append(dates[step])
            # indexes.append(step)
            step+=randint(3, 6)
        except Exception as e: 
            print(e)
            break
    return [{'date': d, 'amnt': roundup(randint(85000, 99980))} for d in l]

def get_forcasting_tab(request_json): 
    res = {}
    if request_json and 'atmId' in request_json:
        res['tab'] = 'F'
        res['tabName'] = 'Forcasting'
        res['testPreformance'] = get_test_performance(request_json['atmId'])
        res['prediction'] = get_prediction(request_json['atmId'], 7)
        res['featureImportances'] = get_feature_importances()
        top_atms_by_rmse_df = pd.DataFrame.from_dict(get_top_atms_by_rmse())
        if request_json['atmId'] not in top_atms_by_rmse_df['atmId'].drop_duplicates().tolist():
            top_atms_by_rmse_df = top_atms_by_rmse_df.append({
                'atmId': request_json['atmId'],
                'rmse': float(res['testPreformance']['metadata']['rmse'])
            }, ignore_index=True)
        res['topAtmsByRMSE'] = top_atms_by_rmse_df.to_dict('records')

    return res


def get_historical_tab(request_json): 
    res = {}
    if request_json and 'atmId' in request_json and 'from' in request_json and 'until':
        start = datetime.strptime(request_json['from'], '%Y-%m-%d').date()
        end = datetime.strptime(request_json['until'], '%Y-%m-%d').date()
        res['tab'] = 'HA'
        res['tabName'] = 'Historical Analysis'
        res['dailyWithdrAmnt'] = get_daily_withr_amnt(start, end, request_json['atmId'])
        res['totalDailyWithdrAmnt'] = get_total_withdr_amnt(start, end)
        res['yearlyWithdrAmnt'] = get_yearly_withdr_amnt(request_json['atmId'])
        res['topAtmsByWithdrAmnt'] = get_top_atms_by_withr_amnt(request_json['atmId'])
        res['metadata'] = get_ha_metadata(request_json['atmId'])
        
    return res

def get_cash_management_tab(request_json): 
    res = {}
    if request_json and 'atmId' in request_json: 
        atm_capacity = 100000
        data = pd.read_csv('./data/fulfill_atms.csv')
        data = data[data.atmId == request_json['atmId']].reset_index(drop=True)
        res['tab'] = 'CM'
        res['tabName'] = 'Cash Management'
        res['replenishmentDates'] = data[
            (data.money_fulfilled != 0)&
            (data.date>=request_json['from']) & 
            (data.date<=request_json['until'])
        ].sort_values(by=['date']).reset_index(drop=True).to_dict('records')
        res['nextFulfillDates'] = fake_dates()    
        res['metadata'] = {
            'remainingAmntLive': roundup(get_remaining(data, atm_capacity)), 
            'nextFulfillDate': res['nextFulfillDates'][0]
        }
        
    return res

def init_filters(): 
    init = fetch_data.fetch_data_df(f'''
        SELECT * 
        FROM `atms_ai_models.trained_atms_metadata`;
    ''')
    init['atmId'] = init['atmId'].astype(int)
    return init.to_dict('records')


def get_exepnocash(request_json): 
    atms_positions = fetch_data.fetch_data_df(f'''
        SELECT * 
        FROM `atms_ai_models.trained_atms_v2`;
    ''')
    atms_positions['remainingAmntLive'] = atms_positions['atmId'].map(lambda x: roundup(randint(1500, 8000)))
    # atm_capacity = 100000
    # data = pd.read_csv('./data/fulfill_atms.csv')
    # result = pd.DataFrame()
    # for i, row in atms_positions.iterrows():
    #     # data = data[data.atmId == int(row['atmId'])].reset_index(drop=True)
    #     row['remainingAmntLive'] = roundup(randint(1500, 8000)) # roundup(get_remaining(data, atm_capacity))
    #     result = result.append(row, ignore_index=True)
    return atms_positions.to_dict('records')


@functions_framework.http
def get_data(request):
    window = 7
    request_json = request.get_json(silent=True)
    request_args = request.args
    # Set CORS headers for the preflight request
    if request.method == 'OPTIONS':
        # Allows GET requests from any origin with the Content-Type
        # header and caches preflight response for an 3600s
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': '*',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }

        return ('', 204, headers)

    # Set CORS headers for the main request
    headers = {
        'Access-Control-Allow-Methods': '*',
        'Access-Control-Allow-Origin': '*'
    }
    

    # if request_json and 'name' in request_json:
    #     name = request_json['name']
    if request_args:
        if 'tab' in request_args:
            tabs = request_args['tab']
            res = {}
            for tab in tabs.split(','):
                if tab.upper() == 'F':
                    res['F'] = get_forcasting_tab(request_json)
                elif tab.upper() == 'HA': 
                    res['HA'] = get_historical_tab(request_json)
                elif tab.upper() == 'CM': 
                    res['CM'] = get_cash_management_tab(request_json)
                elif tab.upper() == 'EC':
                    res['EC'] = get_exepnocash(request_json)
                else: 
                    return ('Please check your params!.', 422, headers)
            return (res, 200, headers)
        
    else:
        res = init_filters()
        return (res, 200, headers)

    return ('Please check your params!.', 422, headers)


