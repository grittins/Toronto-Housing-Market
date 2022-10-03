import pandas as pd
import os

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