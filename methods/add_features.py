import pandas as pd
import numpy as np
from google.cloud import bigquery
import datetime
from sqlalchemy import text 
from sqlalchemy import create_engine 
import datetime
from methods import fetch_data
    
def add_date_features(df):
    df.date = pd.to_datetime(df.date)
    df['year'] = df.date.dt.year
    df['month'] = df.date.dt.month
    df['dayofweek'] = df.date.dt.dayofweek
    df['day'] = df.date.dt.day
    df['dayofyear'] = df.date.dt.dayofyear
    df['weekend'] = df.dayofweek.apply(lambda x: 0 if x <5 else 1)
    df['season'] = df.month % 12 // 3 

    return df

# check if refdate is general holiday or weekend
def is_bank_holiday(refdate, holidays=None):
    holidays = fetch_data.fetch_data_df("SELECT * FROM `atms_ai_models.argies`;")
    holidays.argies =pd.to_datetime(holidays.argies, format='%d/%m/%Y')  
    refdate = pd.to_datetime(refdate)
    refdate_weekday = refdate.weekday()

    return 1 if ((refdate in holidays.argies.to_list()) | ((refdate_weekday) in [5,6])) else 0

def add_pensions_features(df, exclude_pension_name=None):
    pensions_hist = fetch_data.fetch_data_df("SELECT * FROM `atms_ai_models.pensions`;")
    pensions = fetch_data.fetch_data_df("SELECT * FROM `atms_ai_models.Events_20190701`;")
    pensions = pd.concat([pensions_hist, pensions])
    pensions = pensions.melt(var_name='pension_type', value_name='pension_date')
    pensions  = pensions[~pensions.pension_date.isna()] 
    pensions['pension_date'] = pd.to_datetime(pensions.pension_date).dt.date

    # add forward pensions (fake: for inference purposes)
    future_pensions_sql="""
        WITH T1 AS 
        (SELECT * FROM `atms_ai_models.pensions`
        UNION ALL
        SELECT * FROM `atms_ai_models.Events_20190701`)
        ,T2 AS 
        (SELECT 'ika_dimosio' AS pension_type,MAX(ika_dimosio) as pension_date FROM T1 
        UNION ALL
        SELECT 'oaee_oga' AS pension_type,MAX(oaee_oga) as pension_date FROM T1 
        UNION ALL
        SELECT 'nat_kean' AS pension_type,MAX(nat_kean) as pension_date FROM T1 
        UNION ALL
        SELECT 'eteaep' AS pension_type,MAX(eteaep) as pension_date FROM T1  
        ),
        T3 AS
        (
        SELECT pension_type,DATE_ADD(DATE(pension_date), INTERVAL 1 MONTH) FROM T2
        UNION ALL
        SELECT pension_type,DATE_ADD(DATE(pension_date), INTERVAL 2 MONTH) FROM T2
        ),
        T5 AS 
        (SELECT 'ika_dimosio' AS pension_type,DATE(ika_dimosio) as pension_date FROM T1 WHERE ika_dimosio IS NOT NULL
        UNION ALL
        SELECT 'oaee_oga' AS pension_type,DATE(oaee_oga) as pension_date FROM T1 WHERE oaee_oga IS NOT NULL
        UNION ALL
        SELECT 'nat_kean' AS pension_type,DATE(nat_kean) as pension_date FROM T1 WHERE nat_kean IS NOT NULL
        UNION ALL
        SELECT 'eteaep' AS pension_type,DATE(eteaep) as pension_date FROM T1 WHERE eteaep IS NOT NULL),
        T6 AS 
        (SELECT * FROM T5
        UNION ALL 
        SELECT * FROM T3),
        T7 AS 
        (SELECT *,LEFT(CAST(T6.pension_date AS STRING),7) AS pension_ym 
        FROM T6),
        T8 AS  
        (SELECT pension_type,pension_ym,count(*) AS cnt FROM T7 GROUP BY pension_type,pension_ym),
        T9 AS 
        (SELECT T7.*,cnt as pension_month_cnt FROM T7 JOIN T8 ON T7.pension_type=T8.pension_type AND T7.pension_ym=T8.pension_ym),
        T10 AS 
        (SELECT *,ROUND(1.0/pension_month_cnt,2) AS pension_month_ratio,'today' as date_offset FROM T9),
        NEXTWDAY AS 
        (select pension_date,MIN(nd.day) AS odate from T10
        JOIN atms_ai_models.np_dates nd on T10.pension_date<nd.day
        group by pension_date),
        PREWDAY AS 
        (select pension_date,MAX(nd.day) AS odate from T10
        JOIN atms_ai_models.np_dates nd on T10.pension_date>nd.day
        group by pension_date),
        NEXTPENSION AS
        (SELECT pension_type,odate as pension_date,pension_ym,pension_month_cnt,pension_month_ratio,'tomorrow' as date_offset
        FROM T10 JOIN NEXTWDAY ON T10.pension_date=NEXTWDAY.pension_date),
        PREVPENSION AS
        (SELECT pension_type,odate as pension_date,pension_ym,pension_month_cnt,pension_month_ratio,'yesterday' as date_offset
        FROM T10 JOIN PREWDAY ON T10.pension_date=PREWDAY.pension_date)
        SELECT * FROM T10
        UNION ALL
        select * from NEXTPENSION
        UNION ALL 
        select * from PREVPENSION;
    """

    # bq = bigquery.Client(project = 'exepnoproject')  
    pensions= fetch_data.fetch_data_df(future_pensions_sql)
    pensions['pension_date'] = pd.to_datetime(pensions.pension_date).dt.date

    pensions_piv = pensions.pivot(index='pension_date', values=['pension_month_ratio'], columns=['pension_type','date_offset'])
    pensions_piv = pensions_piv.fillna(0)
    pensions_piv.columns = ['_'.join(col[1:]) for col in pensions_piv.columns.values]
    pensions_piv = pensions_piv.reset_index()

    df['date']=pd.to_datetime(df.date).dt.date
    df = df.merge(pensions_piv, left_on = 'date', right_on = 'pension_date', how='left')
    df = df.drop(columns='pension_date') 
    pensions_name = ['ika_dimosio', 'oaee_oga', 'nat_kean', 'eteaep']
    if (exclude_pension_name):
        exclude_pension_cols = [exclude_pension_name +'_' +col for col in ['today','tomorrow','yesterday']]
        df[exclude_pension_cols] = 0

    df['pensions_today_cnt'] = df[[pension +'_today' for pension in pensions_name]].sum(axis=1)
    df['pensions_yesterday_cnt'] = df[[pension +'_yesterday' for pension in pensions_name]].sum(axis=1)
    df['pensions_tomorrow_cnt'] = df[[pension +'_tomorrow' for pension in pensions_name]].sum(axis=1)

    #pensions = pd.concat([pensions, future_pensions])
    return df


def find_yesterday(day, month):
    day_str = str(int(day))
    month_str = str(int(month))

    date_str = day_str + '-' + month_str
    date =  datetime.datetime.strptime(date_str, '%d-%m')
    yesterday = date - datetime.timedelta(days =1)

    return (yesterday.day, yesterday.month)


def find_tomorrow(day, month):
    day_str = str(int(day))
    month_str = str(int(month))

    date_str = day_str + '-' + month_str
    date =  datetime.datetime.strptime(date_str, '%d-%m')
    tomorrow = date + datetime.timedelta(days =1)

    return (tomorrow.day, tomorrow.month)

def add_local_holidays_features(df):  
    local_holidays = fetch_data.fetch_data_df(query = "SELECT * FROM `atms_ai_models.local_holidays`;")
    local_holidays = local_holidays[local_holidays.responsibleBranchCode != '2245_old']
    local_holidays.responsibleBranchCode = pd.to_numeric(local_holidays.responsibleBranchCode)
    local_holidays.dropna(inplace=True)

    local_holidays[['holiday_tomorrow_day', 'holiday_tomorrow_month']] = local_holidays.apply(lambda x: pd.Series(find_yesterday(x.day, x.month)) , axis=1)
    local_holidays[['holiday_yesterday_day', 'holiday_yesterday_month']] = local_holidays.apply(lambda x: pd.Series(find_tomorrow(x.day, x.month)) , axis=1)


    df = df.merge(local_holidays[['responsibleBranchCode', 'day','month']], \
                            on=['responsibleBranchCode', 'day', 'month'] ,\
                            indicator = 'is_holiday_ind', \
                            how='left')
    
    df.loc[df.is_holiday_ind == 'both' ,'is_holiday'] = 1 

    if (df[df.is_holiday_ind == 'both'].shape[0] > 0):
        df = df.merge(local_holidays[['responsibleBranchCode', 'holiday_tomorrow_day','holiday_tomorrow_month']], \
                                    left_on=['responsibleBranchCode', 'day', 'month'], right_on=['responsibleBranchCode', 'holiday_tomorrow_day', 'holiday_tomorrow_month'],\
                                    indicator = 'is_holiday_tomorrow_ind',\
                                    how='left')
        df.loc[df.is_holiday_tomorrow_ind == 'both' ,'is_holiday_tomorrow'] = 1 
        df.drop(columns=['is_holiday_tomorrow_ind',  'holiday_tomorrow_day', 'holiday_tomorrow_month'], inplace=True)

        df = df.merge(local_holidays[['responsibleBranchCode', 'holiday_yesterday_day','holiday_yesterday_month']], \
                                    left_on=['responsibleBranchCode', 'day', 'month'], right_on=['responsibleBranchCode', 'holiday_yesterday_day', 'holiday_yesterday_month'], \
                                    indicator = 'is_holiday_yesterday_ind',\
                                    how='left')
        df.loc[df.is_holiday_yesterday_ind == 'both' ,'is_holiday_yesterday'] = 1 
        df.drop(columns=['is_holiday_yesterday_ind','holiday_yesterday_day', 'holiday_yesterday_month'],inplace=True)

    del df['is_holiday_ind']

    return df


def add_general_holidays_features(df):
    # holidays=pd.read_csv(f'{MNT_PATH}{HOLIDAYS_FILE}', delimiter=';')
    holidays = fetch_data.fetch_data_df("SELECT * FROM `atms_ai_models.argies`;")
    # holidays.argies = pd.to_datetime(holidays.argies, format='%d/%m/%Y')
    holidays.argies = pd.to_datetime(holidays.argies, format='%d/%m/%Y')
    holidays['holiday_yesterday']= holidays.argies + datetime.timedelta(days=1)
    holidays['holiday_tomorrow']= holidays.argies - datetime.timedelta(days=1)

    df = df.merge(holidays[['argies']], left_on='date', right_on='argies', how='left')
    df.loc[~df.argies.isna(), 'is_holiday'] =1
    del df['argies']

    df = df.merge(holidays[['holiday_yesterday']], left_on='date', right_on='holiday_yesterday', how='left')
    df.loc[~df.holiday_yesterday.isna() , 'is_holiday_yesterday'] =1
    del df['holiday_yesterday']

    df = df.merge(holidays[['holiday_tomorrow']], left_on='date', right_on='holiday_tomorrow', how='left')
    df.loc[~df.holiday_tomorrow.isna() , 'is_holiday_tomorrow'] =1
    del df['holiday_tomorrow']

    return df


def add_holidays_features(df):
    #initialize columns
    df['is_holiday'] = 0
    df['is_holiday_tomorrow'] = 0
    df['is_holiday_yesterday'] = 0

    df = add_general_holidays_features(df)
    df = add_local_holidays_features(df)
    return df

def add_is_islandic_feaure(df, summer_winter_ratio_threshold = 100):    
    #exclude holidays / pensions
    df_island  = df[df.pensions_today_cnt + df.pensions_yesterday_cnt + df.pensions_tomorrow_cnt + df.is_holiday == 0]

    #per atm & season find statitics to check seasonality
    statistics_per_atm_season =  df_island.groupby(['atmId','season'])['disbursement_amnt'].describe()
    statistics_per_atm_season = statistics_per_atm_season.reset_index()
    statistics_per_atm_season = statistics_per_atm_season.loc[statistics_per_atm_season.season.isin([0,2]) , ['atmId','season','mean','50%']]
    statistics_per_atm_season.columns = ['atmId','season','mean','median']
    statistics_per_atm_season  = statistics_per_atm_season.pivot(index='atmId', columns='season', values=['mean','median'])
    statistics_per_atm_season.columns = ['mean_winter', 'mean_summer', 'median_winter', 'median_summer'] 
    statistics_per_atm_season['summer_winter_ratio'] = (statistics_per_atm_season.median_summer / statistics_per_atm_season.median_winter - 1)  * 100

    #islandic atms are those which have summer_winter_ratio > summer_winter_ratio_threshold
    islandic_atms = statistics_per_atm_season.loc[statistics_per_atm_season.summer_winter_ratio > summer_winter_ratio_threshold].index.tolist()
    df['is_islandic'] = 0
    df.loc[df.atmId.isin(islandic_atms),'is_islandic'] =1

    return df

#dummy approach for identifying the payment days 
def add_is_payment_feature(df):
    df['is_payment'] = 0
    df.loc[(df.day>=24) & (df.day <= 31),'is_payment'] =1

    return df

def read_data_of_cluster(clusterno = None, datafile= None): 

    bq = bigquery.Client(project = 'exepnoproject')
    df = bq.query(query = "SELECT * FROM `atms_ai_models.atm_trans_OffSite`;").to_dataframe()
    
    #read transactions of onsite-ttw atms that were replaced from offsite
    df_from_ttw = get_ttw_ts()
    
    #concat dfs
    df = pd.concat([df, df_from_ttw])
    atms_df, atmcodes_df = read_atms_data()
    
    df = df.merge(atmcodes_df, on='code')\
            .merge(atms_df, on='atmId')
    
    df['date']= pd.to_datetime(df['date'])
    return df


def add_features(atmId, forcasting_window): 
    start_date = datetime.datetime.now() + datetime.timedelta(days=1)
    end_date = start_date + datetime.timedelta(days=forcasting_window-1)
    pred_data = pd.DataFrame()
    pred_data['date'] = pd.date_range(start=start_date.date(), end=end_date.date(), freq='1D')
    pred_data['atmId'] = int(atmId)
    atms = fetch_data.fetch_data_df(f"SELECT * FROM `atms_ai_models.atms` WHERE AtmId = {int(atmId)};")
    atms = atms.rename(columns={c: c[0].lower() + c[1:] for c in list(atms.columns)})
    pred_data = pd.merge(pred_data, atms, how='left', on=['atmId'])
    pred_data = add_date_features(pred_data) 
    pred_data = add_holidays_features(pred_data)
    pred_data = add_pensions_features(pred_data)
    pred_data = add_is_payment_feature(pred_data)
    pred_data['is_islandic'] = 0
    # pred_data = add_is_islandic_feaure(pred_data)
    independent_vars = {
        'atmId' : 'categorical',  
        'kioskId' : 'categorical',   
        'locationTypeId' : 'categorical', 
        'municipalityId' : 'categorical',
        'prefectureId' : 'categorical', 
        'year' : 'numerical', 
        'month' : 'categorical',
        'dayofweek' : 'categorical', 
        'day' : 'categorical',
        'weekend' : 'categorical', 
        'season' : 'categorical', 
        'ika_dimosio_today' : 'numerical', 
        'oaee_oga_today' : 'numerical',
        'nat_kean_today' : 'numerical', 
        'eteaep_today' : 'numerical', 
        'ika_dimosio_yesterday' : 'numerical',
        'oaee_oga_yesterday' : 'numerical', 
        'nat_kean_yesterday' : 'numerical', 
        'eteaep_yesterday' : 'numerical',
        'ika_dimosio_tomorrow' : 'numerical', 
        'oaee_oga_tomorrow' : 'numerical', 
        'nat_kean_tomorrow' : 'numerical',
        'eteaep_tomorrow' : 'numerical',  
        'is_holiday' : 'categorical', 
        'is_holiday_tomorrow' : 'categorical',
        'is_holiday_yesterday': 'categorical',
        'is_islandic': 'categorical', 
        'is_payment': 'categorical'
    }
    pred_data = pred_data[['date']+list(independent_vars.keys())]

    pred_data.set_index('date', inplace=True)
    pred_data['kioskId'] = pred_data['kioskId'].astype(str)
    pred_data['kioskId'] = pred_data['kioskId'].str.replace('null', '0')
    pred_data['kioskId'] = pred_data['kioskId'].astype(float)
    pred_data.fillna(0, inplace=True)
    return pred_data

