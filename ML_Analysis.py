import pandas as pd
import os
import plotly.express as px

target = ['Average Price']
df_all = pd.read_csv(os.path.join(os.getcwd(),'Team1_Project','Resources','GTA_HomePrice_History.csv'),index_col=0)

def frame_format(df_in,float_cols,date_cols):

    df = df_in.copy()

    for col in float_cols:
        for s in ['$', ',', '%', ' ']:
            df.loc[:, col] = df.loc[:,col].str.replace(s,'')

        df.loc[:,col] = df.loc[:,col].astype('float')
    for col in date_cols:
        df.loc[:,col] = df.loc[:,col].apply(lambda x: pd.Period(x))
    #add in year and quarter cols, drop original date column
    df['Year'] = df.Date.dt.year
    df['Quarter'] = df.Date.dt.quarter
    return df

#cleaning raw data (type converstion and symbol removal from strings
df_all = frame_format(df_all,['Average Price','Average SP/LP'],['Date'])
df_all.set_index('Date',inplace=True)
#df_all.drop('Date',inplace=True, axis=1)
#filter dataset to toronto only, require avg price to >0, and take only community/date groups that have
#more than 30 quarters of data available
df_all_toronto = df_all.loc[df_all.Area=='Toronto',:]
df_all_toronto = df_all_toronto.loc[df_all_toronto['Average Price']>0,:]

df_all_toronto.notna()



element_group_sizes = df_all_toronto.groupby(['Community','Building_Type']).size()>30
element_group_sizes=element_group_sizes[element_group_sizes==1]
#sized groups
grps=tuple(zip(element_group_sizes.reset_index().iloc[:,0].to_list(),element_group_sizes.reset_index().iloc[:,1].to_list()))
element_group_sizes = df_all_toronto.groupby(['Community','Building_Type']).size()>30
df_all_toronto_clean=pd.concat([df_all_toronto.groupby(['Community','Building_Type']).get_group(x) for x in grps])

df_all_toronto_clean.sort_values(by=['Community','Year','Quarter'],inplace=True)
df_all_toronto_clean.drop(['Area','Municipality', 'Dollar Volume'],inplace=True,axis=1)

inflation=pd.read_csv(os.path.join(os.getcwd(),'Team1_Project','Resources','BoCInflation.csv'),header=12,index_col=0)
inflation.index = pd.to_datetime(inflation.index)
inflation.index.dtype
inflation.resample('q')

inflation_q=inflation.resample('q').agg('mean')
mortgage_rates=pd.read_csv(os.path.join(os.getcwd(),'Team1_Project','Resources','BankofCanada-5yearMortgageRates.csv'),index_col=0)
mortgage_rates.index = pd.to_datetime(mortgage_rates.index)
mortgage_rates_q = mortgage_rates.resample('q').agg('mean')
mortgage_rates_q.index=mortgage_rates_q.index.to_period('q')
inflation_q.index=inflation_q.index.to_period('q')

df_all_toronto_clean.join(mortgage_rates_q,on='index',how='inner')
df_all_toronto_clean.join(mortgage_rates_q,how='inner')
df_all_toronto_clean_with_rates=df_all_toronto_clean.join(mortgage_rates_q,how='inner')
df_all_toronto_clean_with_rates=df_all_toronto_clean_with_rates.join(inflation_q['CPI_TRIM'],how='inner')
junction_detached = df_all_toronto_clean_with_rates[(df_all_toronto_clean_with_rates.Community=='Junction Area')&(df_all_toronto_clean_with_rates.Building_Type=='Detached')]
y=junction_detached['Average Price']
jd2=junction_detached.shift(1)
jd2.join(y,how='inner')
jd3=jd2.join(y,how='inner')
jd3['AvgPx_lag1']=jd3['Average Price'].shift(1)
jd3['AvgPx_lag2']=jd3['Average Price'].shift(2)
jd3['AvgPx_lag3']=jd3['Average Price'].shift(3)
jd3['AvgPx_lag4']=jd3['Average Price'].shift(4)
jd3['AvgPx_lag5']=jd3['Average Price'].shift(5)

jd3.drop('Average Price',inplace=True,axis=1)

X_test=jd3['2016Q4':jd3.index.max()]
X_train=jd3[jd3.index.min():'2016Q3']
y2=y[y.index>=jd3.index[0]]
y_test=y2['2016Q4':y2.index.max()]
y_train=y2[y2.index.min():'2016Q3']


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
y_pred_df = pd.DataFrame(index=y_test.index,data=y_pred)
fig = px.scatter(x=y_pred_df,y=y_test)
ally=y_pred_df.join(y_test,how='inner')
ally.columns
fig = px.scatter(ally, x=0,y='Average Price')


#df_encoded = pd.get_dummies(df_all_toronto_clean.drop(columns=['Average Price']))

from sklearn_pandas import DataFrameMapper as dfm


numerical_cols_to_scale=['Sales', 'Dollar Volume',
      'New Listings', 'Average SP/LP', 'Average DOM']
cat_cols_to_encode=['Building_Type','Community']
num_col_no_transform=['Year','Quarter']

mapper_inputs = []

from sklearn.preprocessing import StandardScaler, OneHotEncoder

inputs1 = [([col],StandardScaler(),{'alias':col}) for col in numerical_cols_to_scale]
inputs2 = [([col],OneHotEncoder(),{'alias':col}) for col in cat_cols_to_encode]
inputs3 =  [([col],None,{'alias':col}) for col in num_col_no_transform]
mapper_inputs= [*inputs1,*inputs2,*inputs3]
mapper = dfm(mapper_inputs
             ,df_out=True)

df_encoded = mapper.fit_transform(df_all_toronto_clean.copy())

X = df_encoded.copy()

y = df_all_toronto_clean[target]


from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

### NEED TO ADD LAGGED DATA and use different method
##for time series splitting, hopefully TimeSeriesSplit in sklearn.model_selection
## also need to investigate model scoring and how to forecast
###
###challenge is how to avoid using data we could not have at forecast time
###so need to use pandas to shift dataframe

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=1)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
model.score(X,y)
y_pred = model.predict(X_test)
model.coef_

#
#
# junction=df_all_toronto[df_all_to.Community=='Junction Area']
#
# df_all
#
# df_all_to = df_all_toronto.loc[df_all_toronto['Average Price']>0,:]
#
#
#
# junction.Date=junction.Date.apply(lambda x: pd.Period(x).end_time.date())