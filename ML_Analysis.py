import pandas as pd
import os
import plotly.express as px
from Team1_Project.config import pwd, username, db_name, host, port
import sqlalchemy
from sqlalchemy import create_engine
target = ['average_price']

#df_all = pd.read_csv(os.path.join(os.getcwd(),'Team1_Project','Resources','GTA_HomePrice_History.csv'),index_col=0)
postgres_str = ('postgresql://{username}:{password}@{ipaddress}:{port}/{dbname}'.format(username=username,password=pwd,ipaddress=host,port=port,dbname=db_name))
conn = create_engine(postgres_str)
df_all = pd.read_sql('Select * from home_prices',con=conn)
#inflation = pd.read_sql('Select * from home_prices')
recession = pd.read_sql('Select * from recession_indicator',con=conn)
mortgage_rates = pd.read_sql('Select * from interest_rate',con=conn)
inflation = pd.read_sql('Select * from inflation',con=conn)

# #data preprocessing no longer needed, database source has precleaned data
# def frame_format(df_in,float_cols,date_cols):
#
#     df = df_in.copy()
#
#     for col in float_cols:
#         for s in ['$', ',', '%', ' ']:
#             df.loc[:, col] = df.loc[:,col].str.replace(s,'')
#
#         df.loc[:,col] = df.loc[:,col].astype('float')
#     for col in date_cols:
#         df.loc[:,col] = df.loc[:,col].apply(lambda x: pd.Period(x))
#     #add in year and quarter cols, drop original date column
#     if '_date' in df.columns:
#         df['_year'] = df.Date.dt.year
#         df['quarter'] = df.Date.dt.quarter
#     return df
#
# #cleaning raw data (type converstion and symbol removal from strings)
# df_all = frame_format(df_all,['average_price,'Average_SP_LP'],['_date'])
df_all._date = df_all._date.apply(lambda x: pd.Period(x))
df_all.set_index('_date',inplace=True)
#df_all.drop('_date',inplace=True, axis=1)
#filter dataset to toronto only, require avg price to >0, and take only community/date groups that have
#more than 30 quarters of data available
df_all_toronto = df_all.loc[df_all.area=='Toronto',:]
df_all_toronto = df_all_toronto.loc[df_all_toronto['average_price']>0,:]

#df_all_toronto.notna()



element_group_sizes = df_all_toronto.groupby(['community','building_type']).size()>30
element_group_sizes=element_group_sizes[element_group_sizes==1]
#sized groups
grps=tuple(zip(element_group_sizes.reset_index().iloc[:,0].to_list(),element_group_sizes.reset_index().iloc[:,1].to_list()))
element_group_sizes = df_all_toronto.groupby(['community','building_type']).size()>30
df_all_toronto_clean=pd.concat([df_all_toronto.groupby(['community','building_type']).get_group(x) for x in grps])

df_all_toronto_clean.sort_values(by=['community','building_type','_year','quarter'],inplace=True)
df_all_toronto_clean.drop(['area','municipality', 'dollar_volume','_no'],inplace=True,axis=1)


############Autocorrelation check##################################
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# import matplotlib.pyplot as plt
# # Use the Autocorrelation function
# # from the statsmodel library passing
# # our DataFrame object in as the data
# # Note: Limiting Lags to 50
# acfdata = df_all_toronto_clean[(df_all_toronto_clean.Community=='Junction Area')&(df_all_toronto_clean.Building_Type=='Detached')][['_year','quarter','average_price]]
# acfdata.Date=acfdata.Date.apply(lambda x: pd.Period(x).end_time.date())
# acfdata.set_index(['_date'],inplace=True)
# plot_acf(x=acfdata, lags=25)
# # Show the AR as a plot
# plt.show()
# plot_pacf(x=acfdata, lags=25)
# # Show the AR as a plot
# plt.show()
####################################################################



#
# inflation=pd.read_csv(os.path.join(os.getcwd(),'Team1_Project','Resources','OffGitHub','MLFiles','BoCInflation.csv'),header=12,index_col=0)
# inflation.index = pd.to_datetime(inflation.index)
# inflation_q=inflation.resample('q').agg('mean')
# inflation_q.index=inflation_q.index.to_period('q')
#
# mortgage_rates=pd.read_csv(os.path.join(os.getcwd(),'Team1_Project','Resources','BankofCanada-5yearMortgageRates.csv'),index_col=0)
# mortgage_rates.index = pd.to_datetime(mortgage_rates.index)
# mortgage_rates_q = mortgage_rates.resample('q').agg('mean')
# mortgage_rates_q.index=mortgage_rates_q.index.to_period('q')
#
# recession = pd.read_csv(os.path.join(os.getcwd(),'Team1_Project','Resources','CanadaRecessionIndicator.csv'),header=0,index_col=0,dtype={'CANRECDM':float})
# recession_q = recession.copy()
# recession_q.index=pd.to_datetime(recession_q.index)

###NEED TO ENSURE SERIES IS FLOAT TYPE - initial import had '.' in 2022Q3 row

# recession_q.index = recession_q.index.to_period('q')

#df_all_toronto_clean_with_rates=df_all_toronto_clean.join(mortgage_rates_q,how='inner')
idx = df_all_toronto_clean.index
df_all_toronto_clean_with_rates=df_all_toronto_clean.\
    merge(mortgage_rates,right_on='year_quarter_key',left_on='year_quarter_key',how='left')
#df_all_toronto_clean_with_rates=df_all_toronto_clean_with_rates.join(inflation_q['CPI_TRIM'],how='inner')

df_all_toronto_clean_with_rates=df_all_toronto_clean_with_rates.\
    merge(inflation,right_on='year_quarter_key',left_on='year_quarter_key',how='left')
#df_all_toronto_clean_with_rates=df_all_toronto_clean_with_rates.join(recession_q['CANRECDM'],how='inner')
df_all_toronto_clean_with_rates=df_all_toronto_clean_with_rates.\
    merge(recession[['year_quarter_key','canrecdm']],right_on='year_quarter_key',left_on='year_quarter_key',how='left')

df_all_toronto_clean_with_rates.set_index(idx,inplace=True)

from sktime.forecasting.naive import NaiveForecaster
import sktime
from sktime.forecasting.base import ForecastingHorizon
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError

X = df_all_toronto_clean_with_rates.reset_index().set_index(['community','building_type','_date'])[['average_price','avg_five_year_rates','CPI_TRIM','canrecdm']]
X.sort_index(inplace=True)
y = X['average_price'].to_frame()

cutoff='2018Q4'
y_test=y.loc[(y.index.get_level_values('_date')>cutoff)]
X_test=y.loc[(X.index.get_level_values('_date')>cutoff)]
X_train=y.loc[(X.index.get_level_values('_date')<=cutoff)]
y_train=y.loc[(y.index.get_level_values('_date')<=cutoff)]
fh = ForecastingHorizon(y_test.index.get_level_values(level=2).unique(), is_relative=False)

forecaster = NaiveForecaster(strategy="last", sp=4)

# step 4: fitting the forecaster
forecaster.fit(y_train, X=X_train, fh=fh)

# step 5: querying predictions
y_pred = forecaster.predict(X=X_test)
y_pred2 = y_pred.reindex_like(y_test)
mape = MeanAbsolutePercentageError(symmetric=False)

mape(y_test,y_pred2)



import plotly.graph_objects as go
def make_plots(y_train,y_pred2,y_test,community_filter,building_filter):

    y_train_plot =y_train.loc[(y_train.index.get_level_values('community')==community_filter)&(y_train.index.get_level_values('building_type')==building_filter)]
    y_pred2_plot =y_pred2.loc[(y_pred2.index.get_level_values('community')==community_filter)&(y_pred2.index.get_level_values('building_type')==building_filter)]
    y_test_plot =y_test.loc[(y_test.index.get_level_values('community')==community_filter)&(y_test.index.get_level_values('building_type')==building_filter)]

    y_train_plot=y_train_plot.reset_index()
    y_train_plot=y_train_plot.set_index('_date')
    y_train_plot=y_train_plot.drop(['community','building_type'],axis=1)

    y_pred2_plot=y_pred2_plot.reset_index()
    y_pred2_plot=y_pred2_plot.set_index('_date')
    y_pred2_plot=y_pred2_plot.drop(['community','building_type'],axis=1)

    y_test_plot=y_test_plot.reset_index()
    y_test_plot=y_test_plot.set_index('_date')
    y_test_plot=y_test_plot.drop(['community','building_type'],axis=1)


    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_train_plot.index.to_timestamp(),y=y_train_plot['average_price']))
    fig.add_trace(go.Scatter(x=y_test_plot.index.to_timestamp(),y=y_test_plot['average_price'],mode='lines'))
    fig.add_trace(go.Scatter(x=y_pred2_plot.index.to_timestamp(),y=y_pred2_plot['average_price'],mode='lines'))
    fig.show(renderer='browser')

make_plots(y_train,y_pred2,y_test,'Junction Area','Semi-detached')

# df_all_toronto_clean_with_rates_dummies = pd.get_dummies(df_all_toronto_clean_with_rates)
# #
# # #prep for facebook prophet
# # def prophet_prep(df):
# #     df.index.name = 'ds'
# #     df.reset_index(inplace=True)
# #     df.rename(columns={'average_price': 'y'}, inplace=True)
# #     cols = [x for x in df.columns.to_list() if x not in ['y','ds']]
# #     cols2 = ['ds','y',*cols]
# #
# #     df = df[cols2]
# #
# #     df['ds']=df['ds'].apply(lambda x: x.end_time)
# #
# #     from prophet import Prophet
# #     m = Prophet()
# #
# #     for idx, c in enumerate(df.columns):
# #         if idx > 1:
# #             m.add_regressor(c)
# #
# #     cutoff = '20161231'
# #     df_train = df[df.ds<=cutoff]
# #     df_test = df[df.ds>cutoff]
# #     df_test=df_test.drop('y',axis=1)
# #
# #     return df_train, df_test, m
#
# # df_train, df_test, m = prophet_prep(df_all_toronto_clean_with_rates_dummies)
# #
# # m.fit(df_train)
# # y_pred=m.predict(df_test)
# # yhat=y_pred['yhat']
# # y = df_all_toronto_clean_with_rates_dummies[df_all_toronto_clean_with_rates_dummies.ds>'20161231']['y']
# # y=y.reset_index()
# # y.drop('index',axis=1,inplace=True)
# # ycomp=pd.concat([y,yhat],axis=1)
# # fig=px.scatter(ycomp)
# # resid=ycomp['y']-ycomp['yhat']
# # fig2=px.line(resid)
# # fig2.show(renderer='browser')
# # fig3=px.scatter(x=ycomp['y'],y=ycomp['yhat'])
# # fig3.show(renderer='browser')
#
# df_train = df_all_toronto_clean_with_rates_dummies[df_all_toronto_clean_with_rates_dummies]
#
#
# cols = df_all_toronto_clean_with_rates_dummies.columns.to_list()
# cols2 = [cols[0],cols[2],cols[1],*cols[3:]]
# df_all_toronto_clean_with_rates_dummies = df_all_toronto_clean_with_rates_dummies[cols2]
# df_all_toronto_clean_with_rates_dummies.rename(columns={'average_price':'y'},inplace=True)
# df_all_toronto_clean_with_rates_dummies['ds']=df_all_toronto_clean_with_rates_dummies['ds'].apply(lambda x: x.end_time)
#
#
#
#
#
# junction_detached = df_all_toronto_clean_with_rates[(df_all_toronto_clean_with_rates.Community=='Junction Area')&(df_all_toronto_clean_with_rates.Building_Type=='Detached')]
# y=junction_detached['average_price']
# #junction_detached.drop('average_price',inplace=True,axis=1)
# jd2=junction_detached.shift(1)
#
# jd3=jd2.join(y,how='inner')
# jd3['AvgPx_lag1']=jd3['average_price'].shift(1)
# jd3['AvgPx_lag2']=jd3['average_price'].shift(2)
# jd3['AvgPx_lag3']=jd3['average_price'].shift(3)
# jd3['AvgPx_lag4']=jd3['average_price'].shift(4)
# jd3['AvgPx_lag5']=jd3['average_price'].shift(5)
#
# jd3.drop('average_price',inplace=True,axis=1)
#
# X_test=jd3['2016Q4':jd3.index.max()]
# X_train=jd3[jd3.index.min():'2016Q3']
# y2=y[y.index>=jd3.index[0]]
# y_test=y2['2016Q4':y2.index.max()]
# y_train=y2[y2.index.min():'2016Q3']
#
#
# from sklearn.linear_model import LinearRegression
# model = LinearRegression()
# model.fit(X_train,y_train)
# y_pred = model.predict(X_test)
# y_pred_df = pd.DataFrame(index=y_test.index,data=y_pred)
# fig = px.scatter(x=y_pred_df,y=y_test)
# ally=y_pred_df.join(y_test,how='inner')
# ally.columns
# fig = px.scatter(ally, x=0,y='average_price')
#
#
# #df_encoded = pd.get_dummies(df_all_toronto_clean.drop(columns=['average_price']))
#
# from sklearn_pandas import DataFrameMapper as dfm
#
#
# numerical_cols_to_scale=['Sales', 'Dollar Volume',
#       'New Listings', 'Average SP/LP', 'Average DOM']
# cat_cols_to_encode=['building_type','community']
# num_col_no_transform=['_year','quarter']
#
# mapper_inputs = []
#
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
#
# inputs1 = [([col],StandardScaler(),{'alias':col}) for col in numerical_cols_to_scale]
# inputs2 = [([col],OneHotEncoder(),{'alias':col}) for col in cat_cols_to_encode]
# inputs3 =  [([col],None,{'alias':col}) for col in num_col_no_transform]
# mapper_inputs= [*inputs1,*inputs2,*inputs3]
# mapper = dfm(mapper_inputs
#              ,df_out=True)
#
# df_encoded = mapper.fit_transform(df_all_toronto_clean.copy())
#
# X = df_encoded.copy()
#
# y = df_all_toronto_clean[target]
#
#
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import train_test_split
#
# ### NEED TO ADD LAGGED DATA and use different method
# ##for time series splitting, hopefully TimeSeriesSplit in sklearn.model_selection
# ## also need to investigate model scoring and how to forecast
# ###
# ###challenge is how to avoid using data we could not have at forecast time
# ###so need to use pandas to shift dataframe
#
# X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=1)
#
# from sklearn.linear_model import LinearRegression
# model = LinearRegression()
# model.fit(X_train,y_train)
# model.score(X,y)
# y_pred = model.predict(X_test)
# model.coef_

#
#
# junction=df_all_toronto[df_all_to.Community=='Junction Area']
#
# df_all
#
# df_all_to = df_all_toronto.loc[df_all_toronto['average_price']>0,:]
#
#
#
# junction.Date=junction.Date.apply(lambda x: pd.Period(x).end_time.date())